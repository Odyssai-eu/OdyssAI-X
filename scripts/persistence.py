"""SQLite persistence for Odysseus runs + sync jobs.

Volatile in-memory dicts (`_active_runs`, `_sync_jobs`) lose state on container
restart, which kills diagnostics value: when an issue happens, the run that
caused it is gone before we can inspect it. This module is a thin
write-through layer: each register/tick/finalize call also persists to SQLite
so we keep history (capped) across restarts.

Design notes:
- Best-effort: any SQLite error logs and continues; the in-memory dicts
  remain the source of truth for "live" data, SQLite is for "what happened
  earlier".
- Single file, single connection, single write lock (sqlite3 is fine for
  the load — we expect <1 write/sec sustained, <10/sec burst).
- Schema kept dumb: each row stores a JSON blob of the full payload. Means
  no migrations when we add fields. We just need (id, type, started_at,
  status) as indexed columns for queries.
- Tick updates are throttled (1/sec per run) to avoid hammering the DB on
  every emitted token.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from typing import Any, Optional

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()
_tick_last: dict[str, float] = {}  # rid → last persisted timestamp
_TICK_INTERVAL = 1.0  # seconds


def _log(msg: str) -> None:
    sys.stderr.write(f"[persistence] {msg}\n")


def init_db(db_path: str) -> bool:
    """Open (or create) the SQLite database at `db_path`. Returns True if
    persistence is now active, False if init failed (caller should treat
    persistence as a no-op in that case)."""
    global _conn
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id           TEXT PRIMARY KEY,
              model        TEXT,
              cluster      TEXT,
              client       TEXT,
              started_at   REAL,
              finished_at  REAL,
              status       TEXT,
              payload      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs(status);

            CREATE TABLE IF NOT EXISTS sync_jobs (
              id           TEXT PRIMARY KEY,
              model        TEXT,
              source       TEXT,
              started_at   REAL,
              finished_at  REAL,
              status       TEXT,
              payload      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_started ON sync_jobs(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_jobs_status  ON sync_jobs(status);
            """
        )
        _conn = conn
        _log(f"db ready at {db_path}")
        return True
    except Exception as e:
        _log(f"init failed ({e}) — persistence disabled")
        _conn = None
        return False


def is_ready() -> bool:
    return _conn is not None


# ──────────────────────────────────────────────────────────────────────────
# Runs
# ──────────────────────────────────────────────────────────────────────────

def persist_run(rid: str, payload: dict, *, force: bool = False) -> None:
    """Upsert a run. `force=True` bypasses the tick throttle (use it for
    register + finalize; tick goes through the throttle)."""
    if _conn is None:
        return
    if not force:
        now = time.monotonic()
        last = _tick_last.get(rid, 0.0)
        if now - last < _TICK_INTERVAL:
            return
        _tick_last[rid] = now
    try:
        with _lock:
            _conn.execute(
                """
                INSERT INTO runs(id, model, cluster, client, started_at,
                                  finished_at, status, payload)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  model       = excluded.model,
                  cluster     = excluded.cluster,
                  client      = excluded.client,
                  started_at  = excluded.started_at,
                  finished_at = COALESCE(excluded.finished_at, runs.finished_at),
                  status      = excluded.status,
                  payload     = excluded.payload
                """,
                (
                    rid,
                    payload.get("model"),
                    payload.get("cluster"),
                    payload.get("client"),
                    payload.get("started_at"),
                    payload.get("finished_at"),
                    payload.get("status"),
                    json.dumps(payload, default=str),
                ),
            )
    except Exception as e:
        _log(f"persist_run({rid}) failed: {e}")


def finalize_run(rid: str, payload: dict) -> None:
    """Mark a run as finished. Adds finished_at + ensures status is terminal."""
    payload = dict(payload)
    payload.setdefault("finished_at", time.time())
    if payload.get("status") in (None, "streaming", "running"):
        payload["status"] = "done"
    persist_run(rid, payload, force=True)
    _tick_last.pop(rid, None)


def recent_runs(limit: int = 100, *, status: Optional[str] = None) -> list[dict]:
    if _conn is None:
        return []
    try:
        with _lock:
            if status:
                cur = _conn.execute(
                    "SELECT payload FROM runs WHERE status=? "
                    "ORDER BY started_at DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cur = _conn.execute(
                    "SELECT payload FROM runs ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                )
            return [json.loads(row[0]) for row in cur.fetchall()]
    except Exception as e:
        _log(f"recent_runs failed: {e}")
        return []


def prune_runs(keep: int = 1000) -> int:
    """Keep only the most recent `keep` rows. Returns rows deleted."""
    if _conn is None:
        return 0
    try:
        with _lock:
            cur = _conn.execute(
                "DELETE FROM runs WHERE id NOT IN ("
                "  SELECT id FROM runs ORDER BY started_at DESC LIMIT ?"
                ")",
                (keep,),
            )
            return cur.rowcount or 0
    except Exception as e:
        _log(f"prune_runs failed: {e}")
        return 0


# ──────────────────────────────────────────────────────────────────────────
# Sync jobs
# ──────────────────────────────────────────────────────────────────────────

def persist_job(jid: str, payload: dict) -> None:
    if _conn is None:
        return
    try:
        with _lock:
            _conn.execute(
                """
                INSERT INTO sync_jobs(id, model, source, started_at, finished_at,
                                       status, payload)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  model       = excluded.model,
                  source      = excluded.source,
                  started_at  = excluded.started_at,
                  finished_at = COALESCE(excluded.finished_at, sync_jobs.finished_at),
                  status      = excluded.status,
                  payload     = excluded.payload
                """,
                (
                    jid,
                    payload.get("model"),
                    payload.get("source"),
                    payload.get("started_at"),
                    payload.get("finished_at"),
                    payload.get("status"),
                    json.dumps(payload, default=str),
                ),
            )
    except Exception as e:
        _log(f"persist_job({jid}) failed: {e}")


def recent_jobs(limit: int = 50) -> list[dict]:
    if _conn is None:
        return []
    try:
        with _lock:
            cur = _conn.execute(
                "SELECT payload FROM sync_jobs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            return [json.loads(row[0]) for row in cur.fetchall()]
    except Exception as e:
        _log(f"recent_jobs failed: {e}")
        return []


def mark_orphans_interrupted() -> int:
    """On startup, any sync_job still 'running' is actually orphaned (the
    container restarted). Mark them as 'interrupted' so the dashboard
    shows the true state."""
    if _conn is None:
        return 0
    try:
        with _lock:
            cur = _conn.execute(
                "UPDATE sync_jobs SET status='interrupted', "
                "  finished_at=COALESCE(finished_at, ?) "
                "WHERE status='running'",
                (time.time(),),
            )
            n = cur.rowcount or 0
            if n:
                _log(f"marked {n} orphaned sync_jobs as interrupted")
            return n
    except Exception as e:
        _log(f"mark_orphans failed: {e}")
        return 0


def load_unfinished_jobs() -> list[dict]:
    """Return interrupted/running jobs so the caller can decide to resume."""
    if _conn is None:
        return []
    try:
        with _lock:
            cur = _conn.execute(
                "SELECT payload FROM sync_jobs "
                "WHERE status IN ('interrupted','running') "
                "ORDER BY started_at DESC"
            )
            return [json.loads(row[0]) for row in cur.fetchall()]
    except Exception as e:
        _log(f"load_unfinished_jobs failed: {e}")
        return []
