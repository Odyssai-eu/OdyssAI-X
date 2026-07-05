# Contributing to Odysseus

Thanks for considering a contribution. This document explains what we welcome, the conventions we follow, and the development setup.

## What we welcome

**Always**:
- Bug fixes with a clear reproduction
- New model family support (chat templates, tool-call parsers, EOS sequences)
- New capability blocks for the `x_odyssai` contract
- Performance work backed by `bench` measurements
- Documentation improvements, especially around operator setup

**Discuss first** (open an issue before a PR):
- New endpoints or breaking changes to existing ones
- New backends beyond JACCL / pipeline / tensor-parallel
- New dependency additions
- Anything that touches the cluster lifecycle (boot, JACCL init, RDMA wiring)

## Development setup

The orchestrator runs in a Docker container. For local hacking on the
Python code, you can also run it directly with the requirements
installed in a venv:

```bash
git clone https://github.com/Odyssai-eu/Odysseus.git
cd Odysseus

# Optional: native venv for running the API without Docker
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Single-node smoke test via Docker
mkdir -p ~/.odysseus
cp config/topology.example.yaml ~/.odysseus/topology.yaml
# edit ~/.odysseus/topology.yaml — for single-node Docker, see AGENTS.md
# step 2/3 on how to reach the host machine from inside the container
# (Remote Login + host.docker.internal).
docker compose up --build
curl http://localhost:8000/admin/version
```

For multi-node JACCL development, see [`AGENTS.md`](AGENTS.md) and
[`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md) — you'll need at
least 2 Apple Silicon hosts on the same LAN, plus a Thunderbolt 5 cable
between them if you want RDMA. TCP ring (`backend: ring` in
`topology.yaml`) works without RDMA hardware for early-stage work.

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) with these scopes:

| Type | When |
|---|---|
| `feat` | New capability surface |
| `fix` | Bug fix |
| `perf` | Measurable speedup with bench evidence |
| `refactor` | Internal change, no behaviour delta |
| `docs` | Documentation only |
| `chore` | Tooling, deps, version bumps |

**Title** ≤ 70 chars, imperative. **Body** explains the *why* (1-3 sentences). Use a HEREDOC for multi-line:

```bash
git commit -m "$(cat <<'EOF'
feat(router): semantic auto-routing add-on

Brief description of why this matters — the symptom or constraint
that drove the change.
EOF
)"
```

**Hard rules**:
- Never `--no-verify` (the pre-commit hooks exist for a reason — if one fails, fix the root cause)
- Never `--amend` after a failed pre-commit hook (the commit didn't happen, `--amend` would touch the previous one)
- Never force-push to `main`
- Stage specific files (`git add path/file`) rather than `git add -A` to avoid accidentally including `.env` or other secrets

## Code style

- **Python** — type hints on public functions, 100-char line target. Run `ruff` if you have it installed; no enforced config yet.
- **HTML / JS in `dashboard.html`** — single-file by design, no build step; keep it small
- **Comments** — explain *why*, not *what*. The code already shows what.

## Cluster-touching changes

Changes that affect cluster lifecycle (load, unload, pool wiring, JACCL init) are sensitive — a regression can wedge a multi-node cluster and require physical reboots. The bar for these:

1. Open an issue first to discuss
2. Add a reproduction note in the issue (which pool topology, which model)
3. Test on at least 2-node before merge
4. Document any new environment variables or config keys in `AGENTS.md` or `docs/GETTING-STARTED.md`

## Reporting bugs

Open an issue with:
- Odysseus version (`curl /admin/version`)
- Hardware (Apple Silicon model, RAM, number of nodes)
- Pool configuration (size, model, backend)
- The exact request that failed + the error response
- Relevant runner logs (`docker logs <container>` filtered to the relevant `req_…` id)

## License

By contributing, you agree your contributions are licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).
