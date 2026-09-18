// rdma_probe — libibverbs-only probe of Apple's Thunderbolt RDMA stack.
// No MLX, no JACCL: dlopen("librdma.dylib") exactly like JACCL does, then
// count how many protection domains / completion queues / queue pairs /
// memory regions one process can create on one device, and whether the
// driver reclaims them after a clean exit or a SIGKILL.
//
//   rdma_probe count <dev>                 create PD/CQ/QP/MR until failure, destroy all
//   rdma_probe pd-count <dev>              allocate PDs until failure
//   rdma_probe cycle <dev> <K>             K x (open,pd,cq,qp,mr,destroy,close)
//   rdma_probe hold <dev> <nqp> <nmr> <sec> [clean|dirty]
//                                          create resources, sleep, exit (clean = destroy first,
//                                          dirty = _exit without destroy; or SIGKILL from outside)
// Build: clang -O1 -Wall -o rdma_probe rdma_probe.c
#include <dlfcn.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAXR 1024
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

#define LOAD(sym, var) do { var = dlsym(h, #sym); if (!var) { fprintf(stderr, "dlsym %s: %s\n", #sym, dlerror()); exit(2);} } while (0)

static void load_verbs(void) {
  void *h = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL);
  if (!h) { fprintf(stderr, "dlopen librdma.dylib failed: %s\n", dlerror()); exit(2); }
  LOAD(ibv_get_device_list, p_get_device_list); LOAD(ibv_get_device_name, p_get_device_name);
  LOAD(ibv_open_device, p_open_device); LOAD(ibv_free_device_list, p_free_device_list);
  LOAD(ibv_close_device, p_close_device); LOAD(ibv_alloc_pd, p_alloc_pd); LOAD(ibv_dealloc_pd, p_dealloc_pd);
  LOAD(ibv_create_cq, p_create_cq); LOAD(ibv_destroy_cq, p_destroy_cq); LOAD(ibv_create_qp, p_create_qp);
  LOAD(ibv_destroy_qp, p_destroy_qp); LOAD(ibv_reg_mr, p_reg_mr); LOAD(ibv_dereg_mr, p_dereg_mr);
}

static struct ibv_context *open_dev(const char *name) {
  int n = 0; struct ibv_device **devs = p_get_device_list(&n);
  struct ibv_context *ctx = NULL;
  for (int i = 0; i < n; i++)
    if (strcmp(name, p_get_device_name(devs[i])) == 0) { ctx = p_open_device(devs[i]); break; }
  p_free_device_list(devs);
  if (!ctx) { fprintf(stderr, "open %s failed (errno=%d %s, %d devices)\n", name, errno, strerror(errno), n); exit(3); }
  return ctx;
}

static struct ibv_qp *make_qp(struct ibv_pd *pd, struct ibv_cq *cq) {
  struct ibv_qp_init_attr a; memset(&a, 0, sizeof a);
  a.send_cq = cq; a.recv_cq = cq; a.cap.max_send_wr = 16; a.cap.max_recv_wr = 16;
  a.cap.max_send_sge = 1; a.cap.max_recv_sge = 1; a.qp_type = IBV_QPT_UC; a.sq_sig_all = 0;
  return p_create_qp(pd, &a);
}

static int make_qps(struct ibv_pd *pd, struct ibv_cq *cq, struct ibv_qp **qps, int cap, int *err) {
  int n = 0; for (; n < cap; n++) { errno = 0; qps[n] = make_qp(pd, cq); if (!qps[n]) { *err = errno; break; } } return n;
}
static int make_mrs(struct ibv_pd *pd, struct ibv_mr **mrs, void **bufs, int cap, size_t sz, int *err) {
  int n = 0; for (; n < cap; n++) { if (posix_memalign(&bufs[n], 16384, sz)) { *err = ENOMEM; break; }
    errno = 0; mrs[n] = p_reg_mr(pd, bufs[n], sz, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!mrs[n]) { *err = errno; free(bufs[n]); break; } } return n;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: see header\n"); return 1; }
  load_verbs();
  const char *mode = argv[1], *dev = argv[2];
  static struct ibv_qp *qps[MAXR]; static struct ibv_mr *mrs[MAXR]; static void *bufs[MAXR]; static struct ibv_pd *pds[MAXR];
  int err = 0;

  if (strcmp(mode, "count") == 0) {
    struct ibv_context *ctx = open_dev(dev);
    struct ibv_pd *pd = p_alloc_pd(ctx); if (!pd) { printf("RESULT dev=%s pd=FAIL errno=%d\n", dev, errno); return 4; }
    struct ibv_cq *cq = p_create_cq(ctx, 64, NULL, NULL, 0); if (!cq) { printf("RESULT dev=%s cq=FAIL errno=%d\n", dev, errno); return 4; }
    int nq = make_qps(pd, cq, qps, 64, &err); int qerr = err; err = 0;
    int nm = make_mrs(pd, mrs, bufs, 512, 512 * 1024, &err); int merr = err;
    printf("RESULT dev=%s qp_max=%d qp_errno=%d(%s) mr_max=%d mr_errno=%d(%s)\n", dev, nq, qerr, strerror(qerr), nm, merr, strerror(merr));
    for (int i = 0; i < nm; i++) { p_dereg_mr(mrs[i]); free(bufs[i]); }
    for (int i = 0; i < nq; i++) p_destroy_qp(qps[i]);
    p_destroy_cq(cq); p_dealloc_pd(pd); p_close_device(ctx);
    return 0;
  }
  if (strcmp(mode, "pd-count") == 0) {
    struct ibv_context *ctx = open_dev(dev);
    int n = 0; for (; n < 256; n++) { errno = 0; pds[n] = p_alloc_pd(ctx); if (!pds[n]) { err = errno; break; } }
    printf("RESULT dev=%s pd_max=%d pd_errno=%d(%s)\n", dev, n, err, strerror(err));
    for (int i = 0; i < n; i++) p_dealloc_pd(pds[i]);
    p_close_device(ctx); return 0;
  }
  if (strcmp(mode, "cycle") == 0) {
    int K = argc > 3 ? atoi(argv[3]) : 100; int k = 0;
    for (; k < K; k++) {
      struct ibv_context *ctx = open_dev(dev);
      struct ibv_pd *pd = p_alloc_pd(ctx); if (!pd) { printf("RESULT cycle=%d pd=FAIL errno=%d\n", k, errno); return 4; }
      struct ibv_cq *cq = p_create_cq(ctx, 64, NULL, NULL, 0); if (!cq) { printf("RESULT cycle=%d cq=FAIL errno=%d\n", k, errno); return 4; }
      struct ibv_qp *qp = make_qp(pd, cq); if (!qp) { printf("RESULT cycle=%d qp=FAIL errno=%d(%s)\n", k, errno, strerror(errno)); return 4; }
      void *b; posix_memalign(&b, 16384, 512 * 1024);
      struct ibv_mr *mr = p_reg_mr(pd, b, 512 * 1024, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
      if (!mr) { printf("RESULT cycle=%d mr=FAIL errno=%d(%s)\n", k, errno, strerror(errno)); return 4; }
      p_dereg_mr(mr); free(b); p_destroy_qp(qp); p_destroy_cq(cq); p_dealloc_pd(pd); p_close_device(ctx);
    }
    printf("RESULT dev=%s cycles_ok=%d\n", dev, k); return 0;
  }
  if (strcmp(mode, "hold") == 0) {
    int nqp = atoi(argv[3]), nmr = atoi(argv[4]), sec = atoi(argv[5]); const char *how = argc > 6 ? argv[6] : "clean";
    struct ibv_context *ctx = open_dev(dev);
    struct ibv_pd *pd = p_alloc_pd(ctx); struct ibv_cq *cq = p_create_cq(ctx, 64, NULL, NULL, 0);
    if (!pd || !cq) { printf("RESULT hold pd/cq FAIL errno=%d\n", errno); return 4; }
    int nq = make_qps(pd, cq, qps, nqp, &err); int nm = make_mrs(pd, mrs, bufs, nmr, 512 * 1024, &err);
    printf("HOLD pid=%d dev=%s qp=%d/%d mr=%d/%d mode=%s\n", getpid(), dev, nq, nqp, nm, nmr, how); fflush(stdout);
    sleep(sec);
    if (strcmp(how, "dirty") == 0) { printf("HOLD dirty _exit\n"); fflush(stdout); _exit(0); }
    for (int i = 0; i < nm; i++) { p_dereg_mr(mrs[i]); free(bufs[i]); }
    for (int i = 0; i < nq; i++) p_destroy_qp(qps[i]);
    p_destroy_cq(cq); p_dealloc_pd(pd); p_close_device(ctx);
    printf("HOLD clean exit\n"); return 0;
  }
  fprintf(stderr, "unknown mode %s\n", mode); return 1;
}
