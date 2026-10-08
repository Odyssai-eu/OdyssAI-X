"""Poison-gate: a model that kills its runner must not be re-injected in a
loop by the recovery itself (3-amigos reboot-loop, R2 final, 2026-10-08).

Sticky per-cluster counter — one full recovery (reboot+reload) per operator
intervention; further leaks reboot (the wired must be freed) but never
reload, the cluster stays degraded with a POISON log naming the served
aliases, until the reset endpoint or a MANUAL load re-arms it. Boottime
honesty: a `failed` label with a changed kern.boottime becomes
confirmed-boottime after the nodes return. No node is reached: every seam
is stubbed.

    .venv/bin/python scripts/test_poison_gate.py
"""

import asyncio
import json
import time
import os
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
json.dump({"repl": {"name": "R", "kind": "replica", "backend": "ring",
                    "models_dir": "/m",
                    "nodes": [{"host": f"n{i}", "ssh": f"admin@198.51.100.{i + 1}", "master": i == 0}
                              for i in range(3)]}},
          open(os.environ["CLUSTER_CONFIG_FILE"], "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

# ---------------------------------------------------------------------------
# Seams / helpers
# ---------------------------------------------------------------------------
FAILS: list[str] = []


def check(label: str, cond: bool) -> None:
    if cond:
        print(f"OK {label}")
    else:
        print(f"FAIL {label}")
        FAILS.append(label)


class FakePool:
    def __init__(self, model: str = "momo", alias: str = "momo"):
        self.model = model
        self.alias = alias

    @property
    def alive_count(self) -> int:
        return 1


# --- captured state ---
logs: list[str] = []
rebooted: list[str] = []
cleared: list[str] = []
reload_calls: list[dict] = []
ssh_exec_calls: list[str] = []


def __reset_state() -> None:
    api._leak_reload_done.clear()
    api._leak_reboot_last.clear()
    api._leak_reboot_in_flight.clear()
    api._leak_reboot_in_flight.discard("repl")
    api._leak_reboot_last.clear()
    api._leak_reboot_in_flight.clear()
    api._JACCL_AUTO_RECOVERY_ENABLED = True
    logs.clear()
    rebooted.clear()
    cleared.clear()
    reload_calls.clear()
    ssh_exec_calls.clear()


def _reset_counters() -> None:
    api._leak_reload_done.clear()
    api._leak_reboot_last.clear()
    api._leak_reboot_in_flight.clear()
    api._leak_reboot_in_flight.discard("repl")
    api._leak_reboot_last.clear()


# --- seam stubs ---
def stub_resolve_host(hid):
    return {"id": hid, "ssh": f"admin@198.51.100.{1 + hash(hid) % 3}"}


def make_reboot_stub(method="failed", boottime_before="BT_A"):
    async def _reboot_one(host):
        rebooted.append(host["id"])
        return {
            "host": host["id"],
            "ssh": host["ssh"],
            "rc": 0,
            "error": None,
            "method": method,
            "boottime_before": boottime_before,
        }

    return _reboot_one


async def stub_wait_reachable(cluster_id, host_ids=None):
    return True


async def stub_probe_wired_gb(host):
    return 0.0


def stub_clear_degraded(cluster_id):
    cleared.append(cluster_id)


def stub_jaccl_log(cluster_id, msg):
    logs.append(str(msg))


async def stub_admin_cluster_load(cluster_id, req):
    reload_calls.append({"cluster": cluster_id, "alias": getattr(req, "alias", "?")})
    return {"loaded": True, "load_s": 1.0}


# _ssh_exec stub driven by a queue of (rc, out, err)
_ssh_exec_queue: list[tuple[int, str, str]] = []


def stub_ssh_exec(ssh, cmd, timeout=8):
    ssh_exec_calls.append(cmd)
    if _ssh_exec_queue:
        rc, out, err = _ssh_exec_queue.pop(0)
        if rc < 0:
            # negative rc → simulate subprocess.TimeoutExpired
            import subprocess

            raise subprocess.TimeoutExpired(cmd, timeout)
        return rc, out, err
    return 0, "", ""


# --- apply seams ---
api._resolve_host = stub_resolve_host
api._ssh_exec = stub_ssh_exec
api._wait_nodes_reachable = stub_wait_reachable
api._probe_wired_gb = stub_probe_wired_gb
api._clear_cluster_degraded = stub_clear_degraded
api._jaccl_log = stub_jaccl_log
api.admin_cluster_load = stub_admin_cluster_load

# Ensure set_pool exists (seed the pool so poison log includes the alias)
if hasattr(api, "set_pool"):
    try:
        api.set_pool("repl", "momo", FakePool())
    except Exception:
        pass
else:
    # Fallback: seed list_pools
    if not hasattr(api, "list_pools") or api.list_pools is None:
        api.list_pools = lambda cid: [("momo", FakePool())]

# ---------------------------------------------------------------------------
# Sweep result factory
# ---------------------------------------------------------------------------
def sweep_result(hosts, wired_warn=True):
    return {"swept": [{"host": h, "wired_warn": wired_warn} for h in hosts]}


# ---------------------------------------------------------------------------
# Case 1: FULL tier — counter 0 → reboots, probe OK, clear degraded, reloads
# ---------------------------------------------------------------------------
def case_full() -> None:
    __reset_state()
    api._reboot_one = make_reboot_stub()
    reqs = [FakePool(model="m", alias="momo")]

    asyncio.run(
        api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), reqs)
    )

    check("1 full: rebooted", rebooted == ["node1"])
    check("1 full: clear_degraded called", cleared == ["repl"])
    check("1 full: reload executed", len(reload_calls) == 1)
    check(
        "1 full: counter now 1",
        api._leak_reload_done.get("repl", 0) == 1,
    )
    check("1 full: in_flight cleared", "repl" not in api._leak_reboot_in_flight)


# ---------------------------------------------------------------------------
# Case 2: HOLD tier — counter 1 → reboots still happen, no clear, no reload
# ---------------------------------------------------------------------------
def case_hold() -> None:
    __reset_state()
    api._leak_reload_done["repl"] = 1
    api._reboot_one = make_reboot_stub()
    reqs = [FakePool(model="m", alias="momo")]

    asyncio.run(
        api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), reqs)
    )

    check("2 hold: rebooted still", rebooted == ["node1"])
    check("2 hold: clear_degraded NOT called", cleared == [])
    check("2 hold: no reload", reload_calls == [])
    check("2 hold: degraded conserved (counter unchanged)",
          api._leak_reload_done.get("repl", 0) == 1)
    hold_logs = [l for l in logs if "[poison]" in l]
    check("2 hold: poison log present", len(hold_logs) >= 1)
    check(
        "2 hold: poison log says NOT reloading",
        any("NOT reloading" in l for l in logs),
    )


# ---------------------------------------------------------------------------
# Case 3: Sticky — in hold, the counter does not increase
# ---------------------------------------------------------------------------
def case_sticky() -> None:
    __reset_state()
    api._leak_reload_done["repl"] = 1
    api._reboot_one = make_reboot_stub()

    asyncio.run(
        api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), [FakePool()])
    )
    asyncio.run(
        api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), [FakePool()])
    )

    check(
        "3 sticky: counter stays 1 (no note_reload in hold)",
        api._leak_reload_done.get("repl", 0) == 1,
    )
    check("3 sticky: no reload ever", reload_calls == [])


# ---------------------------------------------------------------------------
# Case 4: Operator reset — _poison_reset("repl") → back to FULL
# ---------------------------------------------------------------------------
def case_reset() -> None:
    __reset_state()
    api._leak_reload_done["repl"] = 1
    api._reboot_one = make_reboot_stub()
    reqs = [FakePool(model="m", alias="momo")]

    # Enter hold first
    asyncio.run(
        api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), reqs)
    )
    hold_cleared = list(cleared)
    hold_reloads = list(reload_calls)

    # Operator resets (and time passes: the 30-min host cooldown of the
    # hold reboot must not mask the gate semantics — simulate "later")
    if hasattr(api, "_poison_reset"):
        api._poison_reset("repl")
        api._leak_reboot_last.clear()
    elif hasattr(api, "poison_reset"):
        api.poison_reset("repl")
    else:
        # No dedicated reset function in the module — emulate the documented
        # contract: a reset clears the full-recovery counter.
        api._leak_reload_done.pop("repl", None)

    # Clear captures and run again
    cleared.clear()
    reload_calls.clear()
    rebooted.clear()
    logs.clear()

    asyncio.run(
        api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), reqs)
    )

    check("4 reset: was hold before", hold_cleared == [] and hold_reloads == [])
    check("4 reset: back to FULL — clear_degraded called", cleared == ["repl"])
    check("4 reset: reloads executed again", len(reload_calls) == 1)
    check(
        "4 reset: counter back to 1 (re-armed)",
        api._leak_reload_done.get("repl", 0) == 1,
    )


# ---------------------------------------------------------------------------
# Case 5: Boottime honesty — method confirmed-boottime / failed / timeout
# ---------------------------------------------------------------------------
def _run_boottime_case(post_exec_responses, method_in, expected_method):
    """Run the sweep with a stubbed reboot_one + scripted _ssh_exec responses."""
    rebooted.clear()
    logs.clear()
    cleared.clear()
    reload_calls.clear()

    # Stub reboot_one with the given method and boottime_before=BT_A
    async def _reboot_one(host):
        rebooted.append(host["id"])
        return {
            "host": host["id"],
            "ssh": host["ssh"],
            "rc": 0,
            "error": None,
            "method": method_in,
            "boottime_before": "BT_A",
        }

    api._reboot_one = _reboot_one

    # Queue responses for the post-check sysctl read
    _ssh_exec_queue.clear()
    _ssh_exec_queue.extend(post_exec_responses)

    asyncio.run(api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), None))

    # Find the final "reboot methods" log (the second one emitted after the
    # post-reachability re-read).
    methods_logs = [l for l in logs if "reboot methods" in l]
    return methods_logs


def case_boottime() -> None:
    __reset_state()

    # 5a: method="failed", boottime_before="BT_A", post-read="BT_B" → confirmed
    logs_m = _run_boottime_case(
        [(0, "BT_B", "")], method_in="failed", expected_method="confirmed-boottime"
    )
    check(
        "5a boottime: failed+BT_B → confirmed-boottime",
        len(logs_m) >= 2 and "confirmed-boottime" in logs_m[-1],
    )

    # Reset state for next scenario
    __reset_state()

    # 5b: method="failed", post-read="BT_A" (unchanged) → stays failed
    logs_m = _run_boottime_case(
        [(0, "BT_A", "")], method_in="failed", expected_method="failed"
    )
    check(
        "5b boottime: failed+BT_A unchanged → stays failed",
        len(logs_m) >= 2 and "confirmed-boottime" not in logs_m[-1]
        and "failed" in logs_m[-1],
    )

    # Reset state for next scenario
    __reset_state()

    # 5c: method="timeout-likely-rebooting", post-read="BT_B" → stays timeout
    logs_m = _run_boottime_case(
        [(0, "BT_B", "")],
        method_in="timeout-likely-rebooting",
        expected_method="timeout-likely-rebooting",
    )
    check(
        "5c boottime: timeout-likely-rebooting+BT_B → stays timeout",
        len(logs_m) >= 2
        and "timeout-likely-rebooting" in logs_m[-1]
        and "confirmed-boottime" not in logs_m[-1],
    )


# ---------------------------------------------------------------------------
# Case 6: Cooldown — host recently rebooted is excluded (skipped)
# ---------------------------------------------------------------------------
def case_cooldown() -> None:
    __reset_state()
    api._reboot_one = make_reboot_stub()
    now = time.time()
    api._leak_reboot_last["node1"] = now  # just rebooted → within cooldown

    asyncio.run(api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), None))

    check("6 cooldown: host excluded (no reboot)", rebooted == [])
    check(
        "6 cooldown: skipped log emitted",
        any("NOT rebooting again" in l for l in logs),
    )


# ---------------------------------------------------------------------------
# Case 7: Poison log contains the pool alias
# ---------------------------------------------------------------------------
def case_poison_log() -> None:
    __reset_state()
    api._leak_reload_done["repl"] = 1  # hold tier
    api._reboot_one = make_reboot_stub()

    # Seed the pool so list_pools returns the alias
    if hasattr(api, "set_pool"):
        try:
            api.set_pool("repl", "momo", FakePool())
        except Exception:
            pass

    asyncio.run(api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), None))

    poison_logs = [l for l in logs if "[poison]" in l]
    check("7 poison log: present", len(poison_logs) >= 1)
    check(
        "7 poison log: contains alias 'momo'",
        any("momo" in l for l in poison_logs),
    )


# ---------------------------------------------------------------------------
# Run all cases
# ---------------------------------------------------------------------------
case_full()
case_hold()
case_sticky()
case_reset()
case_boottime()
case_cooldown()
case_poison_log()


# ---------------------------------------------------------------------------
# Case 8 (codereview f1): recovery WITHOUT a reload snapshot must NOT arm the
# counter — the unload paths spawn it with reload_reqs=None and never
# re-injected anything; arming there wedged clusters degraded.
# ---------------------------------------------------------------------------
__reset_state()
api.set_pool("repl", "momo", FakePool())
api._leak_reboot_last.clear()
asyncio.run(api._auto_reboot_leaked_nodes("repl", sweep_result(["node1"]), None))
check("8 sans snapshot: pas de reload tenté", len(reload_calls) == 0)
check("8 sans snapshot: clear degraded appelé (auto-guérison conservée)",
      cleared == ["repl"])
check("8 sans snapshot: compteur NON armé", api._leak_reload_done.get("repl", 0) == 0)

# ---------------------------------------------------------------------------
# Case 9 (codereview f2): the keepalive ladder is gated the same way.
# ---------------------------------------------------------------------------
__reset_state()
api._leak_reboot_last.clear()


class _LadderReq:
    node_indices = [0]
    nodes = 1
    alias = "momo"


async def _run_ladder():
    await api._keepalive_recovery_ladder("repl", _LadderReq())


asyncio.run(_run_ladder())
check("9 ladder full: reload exécuté, compteur armé",
      len(reload_calls) == 1 and api._leak_reload_done.get("repl", 0) == 1)
__reset_state()
api._leak_reboot_last.clear()
api._leak_reboot_in_flight.clear()
api._leak_reload_done["repl"] = 1
asyncio.run(_run_ladder())
check("9 ladder hold: AUCUN reload, degraded conservé, reboots bien exécutés",
      len(reload_calls) == 0 and cleared == [] and len(rebooted) > 0)

# ---------------------------------------------------------------------------
# Case 10 (codereview f3/f4): the remaining guards, pinned at the source.
# ---------------------------------------------------------------------------
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")).read()
check("10 source: incrément gardé par if reload_reqs ET l'epoch", "if reload_reqs and _leak_epoch.get(cluster_id, 0) == _poison_epoch0:" in _src)
check("10 source: la ladder est gardée elle aussi",
      "keepalive recovery holds" in _src
      and "if _leak_epoch.get(cluster_id, 0) == _epoch0:" in _src)
check("10 source: /admin/reset nautilus réarme", '_poison_reset("nautilus")' in _src)
check("10 source: la route wrapper (et seulement elle) reset au load manuel", "admin_cluster_load_route" in _src)

if FAILS:
    print("FAILS:")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)

print("all OK")
sys.exit(0)
