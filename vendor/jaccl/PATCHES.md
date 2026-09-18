# OdyssAI patches on top of upstream JACCL (MLX v0.32.2)

Every hunk is marked `OdyssAI patch (PATCHES.md #n)` in the source. Base and
licence: see UPSTREAM.md. Rebuild with `scripts/build-jaccl.sh`, deploy with
`scripts/install-jaccl.sh`. Evidence for each patch was measured with the
libibverbs-only probes in `scripts/jaccl/` (`rdma_probe.c`, `rdma_pair.c`) on
two M3 Ultras over Thunderbolt 5, macOS 26.6.1 (2026-09-18).

## #0 — upstream cherry-pick: ring all_gather direction-1 mirror (MLX #4443, 2026-09-15)
`jaccl/ring_impl.h`, 29 lines. The only JACCL commit after v0.32.2 at the time
of vendoring.

## #1 — peer death is visible: side-channel liveness + progress timeout (MLX #4278)
Files: `rdma.h`, `rdma.cpp`, `jaccl.cpp`, `mesh_impl.h`, `ring_impl.h`, `mesh.cpp`, `ring.cpp`.

Fact: on Apple's Thunderbolt RDMA stack a UC queue pair never reports a dead
peer — no error completion, no event — so a survivor polls forever at 100 % CPU
(`rdma_pair.c` scenarios C/E; MLX #4278). RC queue pairs are not supported
(`ibv_create_qp` errno 102), so no hardware retry-exceeded error exists.

Mechanism: the TCP side channel (`TCPAllGather`) lives as long as the group.
`TCPAllGather::fds()` exposes the socket fds; `Config::get_side_channel()`
hands them to `SideChannel::liveness_fds()`, which `MeshGroup`/`RingGroup` pass
to `MeshImpl`/`RingImpl`. Every completion wait loop (6 in mesh, 4 in ring)
owns a `ProgressGuard` and calls `tick(n > 0)` after each `ibv_poll_cq`. While
no completion arrives the guard, at most every 250 ms, `poll()`s the fds
(POLLIN|POLLHUP, timeout 0) and peeks one byte on POLLIN: HUP/ERR/NVAL or a
0-byte peek = the peer process is gone → `std::runtime_error("[jaccl] peer is
gone: side channel to rank N closed while waiting in <op> …")`. No read, no
write, no lock: safe from several ring wire threads at once. Rank 0 watches
every peer; other ranks watch rank 0, whose throw closes its sockets, so a
non-zero rank's death reaches everyone in two hops. Process death (SIGKILL,
crash, jetsam) is detected in ≤ 1 s because the kernel sends FIN; a cable pull
or a host reboot without FIN falls back to the timeout below.

Backstop: `JACCL_PROGRESS_TIMEOUT_S` (or `MLX_JACCL_PROGRESS_TIMEOUT_S`),
default 600 s, 0 disables. No completion for that long → `"[jaccl] no progress
in <op> for N s … lost frame or wedged peer"`. UC has no retransmission, so a
lost frame otherwise hangs the collective forever. 600 s is deliberately longer
than any legitimate wait (a peer busy with a long prefill).

When JACCL is initialised with a user-supplied all-gather (no TCP coordinator)
`liveness_fds()` is empty and only the timeout applies.

## #2 — diagnostics name the link (MLX #3467, exo #1847)
Files: `rdma.cpp`, `tcp.cpp`.

Facts: on a device whose Thunderbolt port is `PORT_DOWN`, `ibv_open_device`
succeeds but `ibv_alloc_pd` returns NULL — that is every "Couldn't allocate
protection domain" we ever saw. A peer whose IPv4-mapped GID is not reachable
on this link fails RTR with errno 60 (ETIMEDOUT); a zero/wrong GID or a wrong
`sgid_index` fails RTR with errno 22. `TCPSocket::recv` reported a stale errno
(typically "errno=2") when `::recv` returned 0 because the coordinator had
closed the socket — which happens to every other rank when one rank fails
during init, and used to be read as an RDMA error.

Now: `alloc_pd` failure names the device and its port state; RTR failure names
the device, port state, the peer's IPv4 from the GID and decodes errno 60/22;
side-channel EOF says "peer closed the side channel (another rank failed
during init or exited)".

## Not patched, by evidence
- No resource leak across processes: after clean exit, `_exit`, SIGKILL of
  either side and five cumulative crashes the device still offers its 10 queue
  pairs (hard per-device limit, errno 16 on the 11th, shared across processes)
  and a fresh connected session works. The "reboot to reset" story was a
  Thunderbolt link/address problem, not leaked queue pairs.
- No reliable transport over UC: RC is unsupported by the stack and a software
  retransmission layer is a redesign. The timeout turns a lost frame into a
  bounded failure the orchestrator already recovers from.
