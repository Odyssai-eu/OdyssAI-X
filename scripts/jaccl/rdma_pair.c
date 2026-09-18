// rdma_pair — connected-QP lifecycle probe between two Macs (libibverbs only, no MLX).
// Mirrors JACCL: UC queue pair, INIT -> RTR -> RTS, IPv4-mapped GID, MTU 1024, PSN 7.
//   server: rdma_pair <dev> server <port> <iters> <sessions>
//   client: rdma_pair <dev> client <server_ip> <port> <iters> <sessions> [die_session:die_iter]
// Each session: fresh PD/CQ/QP/MRs, side-channel exchange over TCP, <iters> ping-pongs
// (post_recv + post_send, poll with a 3 s wall-clock timeout), clean teardown.
// The client may _exit() abruptly at (die_session, die_iter) to simulate a crashed rank;
// the server then reports what the survivor observes, tears down, and tries the next session.
// Build: clang -O1 -Wall -o rdma_pair rdma_pair.c
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

static struct ibv_device **(*p_get_device_list)(int *);
static const char *(*p_get_device_name)(struct ibv_device *);
static struct ibv_context *(*p_open_device)(struct ibv_device *);
static void (*p_free_device_list)(struct ibv_device **);
static int (*p_close_device)(struct ibv_context *);
static struct ibv_pd *(*p_alloc_pd)(struct ibv_context *);
static int (*p_dealloc_pd)(struct ibv_pd *);
static struct ibv_cq *(*p_create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
static int (*p_destroy_cq)(struct ibv_cq *);
static struct ibv_qp *(*p_create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
static int (*p_destroy_qp)(struct ibv_qp *);
static struct ibv_mr *(*p_reg_mr)(struct ibv_pd *, void *, size_t, int);
static int (*p_dereg_mr)(struct ibv_mr *);
static int (*p_query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
static int (*p_query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
static int (*p_modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
#define LOAD(sym, var) do { var = dlsym(h, #sym); if (!var) { fprintf(stderr, "dlsym %s failed\n", #sym); exit(2);} } while (0)
static void load_verbs(void) {
  void *h = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL);
  if (!h) { fprintf(stderr, "dlopen librdma.dylib failed\n"); exit(2); }
  LOAD(ibv_get_device_list, p_get_device_list); LOAD(ibv_get_device_name, p_get_device_name);
  LOAD(ibv_open_device, p_open_device); LOAD(ibv_free_device_list, p_free_device_list); LOAD(ibv_close_device, p_close_device);
  LOAD(ibv_alloc_pd, p_alloc_pd); LOAD(ibv_dealloc_pd, p_dealloc_pd); LOAD(ibv_create_cq, p_create_cq); LOAD(ibv_destroy_cq, p_destroy_cq);
  LOAD(ibv_create_qp, p_create_qp); LOAD(ibv_destroy_qp, p_destroy_qp); LOAD(ibv_reg_mr, p_reg_mr); LOAD(ibv_dereg_mr, p_dereg_mr);
  LOAD(ibv_query_port, p_query_port); LOAD(ibv_query_gid, p_query_gid); LOAD(ibv_modify_qp, p_modify_qp);
}
static double now_s(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec / 1e6; }
static struct ibv_context *open_dev(const char *name) {
  int n = 0; struct ibv_device **devs = p_get_device_list(&n); struct ibv_context *ctx = NULL;
  for (int i = 0; i < n; i++) if (strcmp(name, p_get_device_name(devs[i])) == 0) { ctx = p_open_device(devs[i]); break; }
  p_free_device_list(devs);
  if (!ctx) { printf("FAIL open_device %s errno=%d\n", name, errno); exit(3); }
  return ctx;
}
struct dest { uint32_t qpn; uint16_t lid; union ibv_gid gid; };
struct sess { struct ibv_context *ctx; struct ibv_pd *pd; struct ibv_cq *cq; struct ibv_qp *qp; struct ibv_mr *smr, *rmr; char *sbuf, *rbuf; struct dest me; };
#define BUF (64 * 1024)
static int g_send_len = 4096, g_recv_len = 4096, g_sgid = 1, g_sleep_ms = 0, g_rc = 0, g_nogid = 0; static const char *g_gidip = NULL;

static int setup(struct sess *s, const char *dev, int session) {
  memset(s, 0, sizeof *s);
  s->ctx = open_dev(dev);
  errno = 0; s->pd = p_alloc_pd(s->ctx); if (!s->pd) { printf("S%d FAIL alloc_pd errno=%d(%s)\n", session, errno, strerror(errno)); return -1; }
  errno = 0; s->cq = p_create_cq(s->ctx, 64, NULL, NULL, 0); if (!s->cq) { printf("S%d FAIL create_cq errno=%d\n", session, errno); return -1; }
  struct ibv_qp_init_attr a; memset(&a, 0, sizeof a);
  a.send_cq = s->cq; a.recv_cq = s->cq; a.cap.max_send_wr = 32; a.cap.max_recv_wr = 4000; a.cap.max_send_sge = 1; a.cap.max_recv_sge = 1;
  a.qp_type = g_rc ? IBV_QPT_RC : IBV_QPT_UC; a.sq_sig_all = 0;
  errno = 0; s->qp = p_create_qp(s->pd, &a); if (!s->qp) { printf("S%d FAIL create_qp errno=%d(%s)\n", session, errno, strerror(errno)); return -1; }
  posix_memalign((void **)&s->sbuf, 16384, BUF); posix_memalign((void **)&s->rbuf, 16384, BUF);
  int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  s->smr = p_reg_mr(s->pd, s->sbuf, BUF, acc); s->rmr = p_reg_mr(s->pd, s->rbuf, BUF, acc);
  if (!s->smr || !s->rmr) { printf("S%d FAIL reg_mr errno=%d\n", session, errno); return -1; }
  struct ibv_qp_attr at; memset(&at, 0, sizeof at);
  at.qp_state = IBV_QPS_INIT; at.port_num = 1; at.pkey_index = 0; at.qp_access_flags = acc;
  int st = p_modify_qp(s->qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (st) { printf("S%d FAIL modify INIT errno=%d(%s)\n", session, st, strerror(st)); return -1; }
  struct ibv_port_attr pa; p_query_port(s->ctx, 1, &pa);
  int found = 0;
  for (int i = 0; i < pa.gid_tbl_len; i++) { union ibv_gid g; if (p_query_gid(s->ctx, 1, i, &g) == 0) {
      if (*(uint64_t *)&g.raw[0] == 0 && *(uint16_t *)&g.raw[8] == 0 && *(uint16_t *)&g.raw[10] == 0xffff) { s->me.gid = g; found = 1; printf("S%d gid_index=%d gid_tbl_len=%d lid=%d\n", session, i, pa.gid_tbl_len, pa.lid); break; } } }
  if (!found) { printf("S%d FAIL no IPv4-mapped GID (gid_tbl_len=%d)\n", session, pa.gid_tbl_len); return -1; }
  s->me.lid = pa.lid; s->me.qpn = s->qp->qp_num; if (g_nogid) memset(&s->me.gid, 0, sizeof s->me.gid); if (g_gidip) { memset(&s->me.gid, 0, sizeof s->me.gid); s->me.gid.raw[10] = 0xff; s->me.gid.raw[11] = 0xff; inet_pton(AF_INET, g_gidip, &s->me.gid.raw[12]); }
  return 0;
}
static int connect_qp(struct sess *s, struct dest *d, int session) {
  struct ibv_qp_attr at; memset(&at, 0, sizeof at);
  at.qp_state = IBV_QPS_RTR; at.path_mtu = IBV_MTU_1024; at.rq_psn = 7; at.dest_qp_num = d->qpn;
  at.ah_attr.dlid = d->lid; at.ah_attr.sl = 0; at.ah_attr.src_path_bits = 0; at.ah_attr.port_num = 1; at.ah_attr.is_global = 0;
  if (d->gid.global.interface_id) { at.ah_attr.is_global = 1; at.ah_attr.grh.hop_limit = 1; at.ah_attr.grh.dgid = d->gid; at.ah_attr.grh.sgid_index = g_sgid; }
  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
  if (g_rc) { at.max_dest_rd_atomic = 1; at.min_rnr_timer = 12; mask |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER; }
  int st = p_modify_qp(s->qp, &at, mask);
  if (st) { printf("S%d FAIL modify RTR errno=%d(%s)\n", session, st, strerror(st)); return -1; }
  memset(&at, 0, sizeof at); at.qp_state = IBV_QPS_RTS; at.sq_psn = 7; mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
  if (g_rc) { at.timeout = 14; at.retry_cnt = 7; at.rnr_retry = 7; at.max_rd_atomic = 1; mask |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC; }
  st = p_modify_qp(s->qp, &at, mask);
  if (st) { printf("S%d FAIL modify RTS errno=%d(%s)\n", session, st, strerror(st)); return -1; }
  return 0;
}
static void teardown(struct sess *s) {
  if (s->smr) p_dereg_mr(s->smr); if (s->rmr) p_dereg_mr(s->rmr); free(s->sbuf); free(s->rbuf);
  if (s->qp) p_destroy_qp(s->qp); if (s->cq) p_destroy_cq(s->cq); if (s->pd) p_dealloc_pd(s->pd); if (s->ctx) p_close_device(s->ctx);
}
// one ping-pong: post recv, post send, wait for 2 completions (timeout s). returns 0 ok, 1 timeout, 2 wc error
static int pingpong(struct sess *s, int iter, double timeout, int *nrecv, int *nsend, int *wcstatus) {
  int st;
  struct ibv_sge ssge = { .addr = (uintptr_t)s->sbuf, .length = (uint32_t)g_send_len, .lkey = s->smr->lkey };
  struct ibv_send_wr swr = { .wr_id = 1, .sg_list = &ssge, .num_sge = 1, .opcode = IBV_WR_SEND, .send_flags = IBV_SEND_SIGNALED }, *bad_s;
  st = ibv_post_send(s->qp, &swr, &bad_s); if (st) { printf("  iter %d post_send failed errno=%d(%s)\n", iter, st, strerror(st)); *wcstatus = -st; return 2; }
  int got = 0; double t0 = now_s(); *nrecv = *nsend = 0; *wcstatus = 0;
  while (got < 2) {
    struct ibv_wc wc[4]; int n = ibv_poll_cq(s->cq, 4, wc);
    for (int i = 0; i < n; i++) { got++; if (wc[i].wr_id >= 1000) (*nrecv)++; else (*nsend)++; if (wc[i].status != IBV_WC_SUCCESS) { printf("  iter %d WC wr_id=%llu opcode=%d status=%d vendor_err=%u byte_len=%u\n", iter, (unsigned long long)wc[i].wr_id, wc[i].opcode, wc[i].status, wc[i].vendor_err, wc[i].byte_len); *wcstatus = wc[i].status; return 2; } }
    if (now_s() - t0 > timeout) return 1;
  }
  return 0;
}
static int xchg(int fd, struct dest *me, struct dest *peer) {
  if (send(fd, me, sizeof *me, 0) != sizeof *me) return -1;
  size_t got = 0; while (got < sizeof *peer) { ssize_t r = recv(fd, (char *)peer + got, sizeof *peer - got, 0); if (r <= 0) return -1; got += r; }
  return 0;
}
int main(int argc, char **argv) {
  if (argc < 6) { fprintf(stderr, "usage: see header\n"); return 1; }
  load_verbs();
  if (getenv("SEND_LEN")) g_send_len = atoi(getenv("SEND_LEN")); if (getenv("RECV_LEN")) g_recv_len = atoi(getenv("RECV_LEN")); if (getenv("SGID_IDX")) g_sgid = atoi(getenv("SGID_IDX")); if (getenv("ITER_SLEEP_MS")) g_sleep_ms = atoi(getenv("ITER_SLEEP_MS")); if (getenv("QP_TYPE") && strcmp(getenv("QP_TYPE"), "RC") == 0) g_rc = 1; if (getenv("FORCE_NO_GID")) g_nogid = 1; if (getenv("FORCE_GID_IP")) g_gidip = getenv("FORCE_GID_IP");
  const char *dev = argv[1]; int is_server = strcmp(argv[2], "server") == 0;
  int port, iters, sessions, die_s = -1, die_i = -1; const char *ip = NULL;
  if (is_server) { port = atoi(argv[3]); iters = atoi(argv[4]); sessions = atoi(argv[5]); }
  else { ip = argv[3]; port = atoi(argv[4]); iters = atoi(argv[5]); sessions = atoi(argv[6]); if (argc > 7) sscanf(argv[7], "%d:%d", &die_s, &die_i); }
  int lfd = -1;
  if (is_server) { lfd = socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = INADDR_ANY };
    if (bind(lfd, (struct sockaddr *)&sa, sizeof sa) || listen(lfd, 4)) { perror("bind/listen"); return 1; } }
  for (int se = 1; se <= sessions; se++) {
    struct sess s; double t0 = now_s();
    if (setup(&s, dev, se)) { teardown(&s); return 4; }
    int fd;
    if (is_server) { fd = accept(lfd, NULL, NULL); if (fd < 0) { perror("accept"); return 1; } }
    else { fd = socket(AF_INET, SOCK_STREAM, 0); struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons(port) }; inet_pton(AF_INET, ip, &sa.sin_addr);
      int tries = 0; while (connect(fd, (struct sockaddr *)&sa, sizeof sa)) { if (++tries > 50) { perror("connect"); return 1; } usleep(200000); close(fd); fd = socket(AF_INET, SOCK_STREAM, 0); } }
    struct dest peer; if (xchg(fd, &s.me, &peer)) { printf("S%d side-channel exchange failed\n", se); teardown(&s); close(fd); return 5; }
    if (connect_qp(&s, &peer, se)) { teardown(&s); close(fd); return 4; }
    // pre-post every receive of the session BEFORE the sync: UC drops a message that
    // arrives before a receive is posted (no RNR), exactly why JACCL pre-posts its recvs.
    for (int it = 1; it <= iters; it++) {
      struct ibv_sge sge = { .addr = (uintptr_t)s.rbuf, .length = (uint32_t)g_recv_len, .lkey = s.rmr->lkey };
      struct ibv_recv_wr rwr = { .wr_id = 1000 + it, .sg_list = &sge, .num_sge = 1 }, *bad_r;
      int st = ibv_post_recv(s.qp, &rwr, &bad_r);
      if (st) { printf("S%d pre-post recv %d failed errno=%d(%s)\n", se, it, st, strerror(st)); teardown(&s); close(fd); return 6; }
    }
    // both sides sync once more over TCP so no one sends before the peer is in RTR
    char ok = 1; send(fd, &ok, 1, 0); recv(fd, &ok, 1, 0);
    int rc = 0, nr = 0, ns = 0, wcs = 0, done = 0;
    for (int it = 1; it <= iters; it++) {
      if (!is_server && se == die_s && it == die_i) { printf("S%d iter %d CLIENT DIES (_exit, no teardown)\n", se, it); fflush(stdout); _exit(0); }
      rc = pingpong(&s, it, 3.0, &nr, &ns, &wcs);
      if (rc) { printf("S%d iter %d %s: recv=%d send=%d wc_status=%d after %.2fs\n", se, it, rc == 1 ? "TIMEOUT" : "WC_ERROR", nr, ns, wcs, now_s() - t0); break; }
      done = it; if (g_sleep_ms) usleep(g_sleep_ms * 1000);
    }
    printf("S%d %s: %d/%d iters ok, qpn=%u peer_qpn=%u, %.2fs; teardown\n", se, rc ? "BROKEN" : "OK", done, iters, s.me.qpn, peer.qpn, now_s() - t0); fflush(stdout);
    close(fd); teardown(&s);
  }
  return 0;
}
