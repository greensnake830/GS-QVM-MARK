#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <windows.h>
#include <commctrl.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include "QVM_cuda.h"

#pragma comment(lib, "comctl32.lib")

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_QUBITS       32
#define MAX_FUSE         512
#define MAX_FUSE_QUBITS  32

#define ID_BTN_START 1001
#define ID_EDIT_LOG  1002
#define WM_UPDATE_UI (WM_USER + 1)
#define WM_FINISH_BENCH (WM_USER + 2)

typedef std::complex<double> cd;
static double now_sec(void) { return omp_get_wtime(); }

typedef struct { uint64_t s[2]; } Rng;
static void rng_seed(Rng *r, uint64_t seed) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; r->s[0] = z ^ (z >> 31);
    z = seed + 0x9E3779B97F4A7C15ULL * 2; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; r->s[1] = z ^ (z >> 31);
    if (r->s[0] == 0 && r->s[1] == 0) r->s[0] = 1;
}
static inline uint64_t rng_next(Rng *r) {
    uint64_t x = r->s[0]; const uint64_t y = r->s[1]; r->s[0] = y; x ^= x << 23; r->s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); return r->s[1] + y;
}
static inline double rng_double(Rng *r) { return (rng_next(r) >> 11) * (1.0 / 9007199254740992.0); }
static double rng_double_wrapper(void *ctx) { return rng_double((Rng *)ctx); }

typedef struct { int active; cd matrix[2][2]; int count; } FuseSlot;
typedef struct {
    int n_qubits, n_clbits; long long dim; cdd *state; int *cbits;
    FuseSlot fuse_slots[MAX_FUSE_QUBITS]; Rng rng;
} QVMContext;

typedef enum {
    I_QREG, I_CREG, I_MEASURE, I_RESET, I_GATE1, I_GATE1_CTRL, I_CX, I_CZ, I_CP, I_SWAP, I_CCX, I_CSWAP, I_GATE2_PAULI
} InstrKind;

typedef struct { InstrKind kind; int q0, q1, q2, c0, c1, has_if; cd m[2][2]; double theta; int sub; } Instr;
typedef struct { Instr *items; int count, cap; } Program;

static void prog_init(Program *p) { p->items = NULL; p->count = p->cap = 0; }
static void prog_free(Program *p) { free(p->items); p->items = NULL; p->count = p->cap = 0; }
static int prog_push(Program *p, Instr ins) {
    if (p->count == p->cap) {
        int nc = p->cap ? p->cap * 2 : 64; Instr *ni = (Instr *)realloc(p->items, nc * sizeof(Instr));
        if (!ni) return 0; p->items = ni; p->cap = nc;
    }
    p->items[p->count++] = ins; return 1;
}

static double noise_bit_flip = 0.0, noise_phase_flip = 0.0, noise_depolar = 0.0, noise_measure = 0.0;
static double t1_per_qubit[MAX_QUBITS], t2_per_qubit[MAX_QUBITS];
static double gate_time_1q = 1.0, gate_time_2q = 2.0, gate_time_measure = 5.0;
static volatile LONG g_progress_done = 0, g_progress_total = 0, g_progress_active = 0;
static double g_progress_t0 = 0.0;
static volatile LONG g_stress_active = 0;

HWND g_hWnd;
HWND g_hEditLog;

static int measure_qubit(QVMContext *ctx, int q);
static void fuse_flush_all(QVMContext *ctx);

static void fuse_init_all(QVMContext *ctx) {
    for (int i = 0; i < MAX_FUSE_QUBITS; i++) {
        ctx->fuse_slots[i].active = 0; ctx->fuse_slots[i].count = 0;
        ctx->fuse_slots[i].matrix[0][0] = 1; ctx->fuse_slots[i].matrix[0][1] = 0; ctx->fuse_slots[i].matrix[1][0] = 0; ctx->fuse_slots[i].matrix[1][1] = 1;
    }
}
static void ctx_init(QVMContext *ctx, uint64_t seed) { memset(ctx, 0, sizeof(*ctx)); fuse_init_all(ctx); rng_seed(&ctx->rng, seed); }
static void ctx_free_state(QVMContext *ctx) {
    if (ctx->state) { gpu_batch_free(ctx->state); ctx->state = NULL; }
    if (ctx->cbits) { free(ctx->cbits); ctx->cbits = NULL; }
    ctx->n_qubits = ctx->n_clbits = ctx->dim = 0;
}
static void ctx_free(QVMContext *ctx) { ctx_free_state(ctx); }

static void ctx_init_state(QVMContext *ctx, int n) { ctx->n_qubits = n; ctx->dim = 1LL << n; ctx->state = gpu_batch_alloc(1, ctx->dim); gpu_batch_init_zero(ctx->state, 1, ctx->dim); }
static void ctx_init_cbits(QVMContext *ctx, int n) { if (ctx->cbits) free(ctx->cbits); ctx->n_clbits = n; ctx->cbits = (int *)calloc(n, sizeof(int)); }

static void apply_1q(QVMContext *ctx, int q, cd m00, cd m01, cd m10, cd m11) { gpu_apply_1q(ctx->state, ctx->dim, q, m00, m01, m10, m11); }
static void apply_1q_ctrl(QVMContext *ctx, int c, int t, cd m00, cd m01, cd m10, cd m11) { gpu_apply_1q_ctrl(ctx->state, ctx->dim, c, t, m00, m01, m10, m11); }
static void apply_cx(QVMContext *ctx, int ctrl, int targ) { gpu_apply_cx(ctx->state, ctx->dim, ctrl, targ); }
static void apply_cz(QVMContext *ctx, int ctrl, int targ) { gpu_apply_cz(ctx->state, ctx->dim, ctrl, targ); }
static void apply_cp(QVMContext *ctx, int ctrl, int targ, double theta) { gpu_apply_cp(ctx->state, ctx->dim, ctrl, targ, theta); }
static void apply_swap(QVMContext *ctx, int a, int b) { gpu_apply_swap(ctx->state, ctx->dim, a, b); }
static void apply_ccx(QVMContext *ctx, int c1, int c2, int targ) { gpu_apply_ccx(ctx->state, ctx->dim, c1, c2, targ); }

static void matmul_2x2(const cd a[2][2], const cd b[2][2], cd out[2][2]) {
    out[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0]; out[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1];
    out[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0]; out[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1];
}
static void fuse_flush_one(QVMContext *ctx, int q) {
    if (!ctx->fuse_slots[q].active || ctx->fuse_slots[q].count == 0) return;
    const cd (*M)[2] = ctx->fuse_slots[q].matrix; apply_1q(ctx, q, M[0][0], M[0][1], M[1][0], M[1][1]); ctx->fuse_slots[q].active = 0; ctx->fuse_slots[q].count = 0;
}
static void fuse_flush_all(QVMContext *ctx) { for (int q = 0; q < ctx->n_qubits; q++) fuse_flush_one(ctx, q); }
static void fuse_add_1q(QVMContext *ctx, int q, cd m00, cd m01, cd m10, cd m11) {
    FuseSlot *s = &ctx->fuse_slots[q];
    if (!s->active) { s->active = 1; s->matrix[0][0] = 1; s->matrix[0][1] = 0; s->matrix[1][0] = 0; s->matrix[1][1] = 1; s->count = 0; }
    cd m[2][2] = {{m00, m01}, {m10, m11}}; cd result[2][2]; matmul_2x2(m, (const cd (*)[2])s->matrix, result);
    s->matrix[0][0] = result[0][0]; s->matrix[0][1] = result[0][1]; s->matrix[1][0] = result[1][0]; s->matrix[1][1] = result[1][1];
    s->count++; if (s->count >= MAX_FUSE) fuse_flush_one(ctx, q);
}

static inline int rand_bit(QVMContext *ctx, double p) { return rng_double(&ctx->rng) < p ? 1 : 0; }
static void inject_noise(QVMContext *ctx, int q) {
    if (noise_bit_flip > 0.0 && rand_bit(ctx, noise_bit_flip)) fuse_add_1q(ctx, q, 0, 1, 1, 0);
    if (noise_phase_flip > 0.0 && rand_bit(ctx, noise_phase_flip)) fuse_add_1q(ctx, q, 1, 0, 0, -1);
    if (noise_depolar > 0.0) {
        double r = rng_double(&ctx->rng);
        if (r < noise_depolar / 3.0) fuse_add_1q(ctx, q, 0, 1, 1, 0);
        else if (r < 2.0 * noise_depolar / 3.0) fuse_add_1q(ctx, q, 0, cd(0,-1), cd(0,1), 0);
        else if (r < noise_depolar) fuse_add_1q(ctx, q, 1, 0, 0, -1);
    }
}

static void apply_t1_t2(QVMContext *ctx, int q, double dt) {
    if (q < 0 || q >= MAX_QUBITS || dt <= 0.0) return;
    double t1 = t1_per_qubit[q], t2 = t2_per_qubit[q];
    if (t1 <= 0.0 && t2 <= 0.0) return;
    if (t1 > 0.0) { double p1 = 1.0 - exp(-dt / t1); if (rng_double(&ctx->rng) < p1) { int r = measure_qubit(ctx, q); if (r == 1) apply_1q(ctx, q, 0, 1, 1, 0); } }
    if (t2 > 0.0) {
        double sigma2 = dt / t2, u1 = rng_double(&ctx->rng), u2 = rng_double(&ctx->rng);
        if (u1 < 1e-300) u1 = 1e-300; double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2); double phi = z * sqrt(sigma2);
        fuse_add_1q(ctx, q, 1, 0, 0, std::exp(cd(0, phi)));
    }
}

static int measure_qubit(QVMContext *ctx, int q) { return gpu_measure(ctx->state, ctx->dim, q, rng_double_wrapper, &ctx->rng); }

static void exec_instr(QVMContext *ctx, const Instr *ins, int verbose) {
    if (ins->has_if) { if (ins->c0 < 0 || ins->c0 >= ctx->n_clbits) return; if (ctx->cbits[ins->c0] != ins->c1) return; }
    switch (ins->kind) {
    case I_QREG: { int n = ins->q0; if (n <= 0 || n > MAX_QUBITS) return; ctx_free_state(ctx); fuse_init_all(ctx); ctx_init_state(ctx, n); break; }
    case I_CREG: ctx_init_cbits(ctx, ins->q0); break;
    case I_GATE1: { int q = ins->q0; fuse_add_1q(ctx, q, ins->m[0][0], ins->m[0][1], ins->m[1][0], ins->m[1][1]); inject_noise(ctx, q); apply_t1_t2(ctx, q, gate_time_1q); break; }
    case I_GATE1_CTRL: { int c = ins->q0, t = ins->q1; fuse_flush_all(ctx); apply_1q_ctrl(ctx, c, t, ins->m[0][0], ins->m[0][1], ins->m[1][0], ins->m[1][1]); inject_noise(ctx, c); inject_noise(ctx, t); apply_t1_t2(ctx, c, gate_time_2q); apply_t1_t2(ctx, t, gate_time_2q); break; }
    case I_CX: { fuse_flush_all(ctx); apply_cx(ctx, ins->q0, ins->q1); inject_noise(ctx, ins->q0); inject_noise(ctx, ins->q1); apply_t1_t2(ctx, ins->q0, gate_time_2q); apply_t1_t2(ctx, ins->q1, gate_time_2q); break; }
    case I_CZ: { fuse_flush_all(ctx); apply_cz(ctx, ins->q0, ins->q1); inject_noise(ctx, ins->q0); inject_noise(ctx, ins->q1); apply_t1_t2(ctx, ins->q0, gate_time_2q); apply_t1_t2(ctx, ins->q1, gate_time_2q); break; }
    case I_CP: { fuse_flush_all(ctx); apply_cp(ctx, ins->q0, ins->q1, ins->theta); inject_noise(ctx, ins->q0); inject_noise(ctx, ins->q1); apply_t1_t2(ctx, ins->q0, gate_time_2q); apply_t1_t2(ctx, ins->q1, gate_time_2q); break; }
    case I_SWAP: { fuse_flush_all(ctx); apply_swap(ctx, ins->q0, ins->q1); inject_noise(ctx, ins->q0); inject_noise(ctx, ins->q1); apply_t1_t2(ctx, ins->q0, gate_time_2q); apply_t1_t2(ctx, ins->q1, gate_time_2q); break; }
    case I_CCX: { fuse_flush_all(ctx); apply_ccx(ctx, ins->q0, ins->q1, ins->q2); inject_noise(ctx, ins->q0); inject_noise(ctx, ins->q1); inject_noise(ctx, ins->q2); apply_t1_t2(ctx, ins->q0, gate_time_2q); apply_t1_t2(ctx, ins->q1, gate_time_2q); apply_t1_t2(ctx, ins->q2, gate_time_2q); break; }
    case I_CSWAP: {
        fuse_flush_all(ctx); int c = ins->q0, a = ins->q1, b = ins->q2;
        apply_cx(ctx, c, b); apply_ccx(ctx, c, b, a); apply_cx(ctx, c, b);
        inject_noise(ctx, c); inject_noise(ctx, a); inject_noise(ctx, b); apply_t1_t2(ctx, c, gate_time_2q); apply_t1_t2(ctx, a, gate_time_2q); apply_t1_t2(ctx, b, gate_time_2q); break;
    }
    case I_GATE2_PAULI: {
        fuse_flush_all(ctx); int a = ins->q0, b = ins->q1; double th = ins->theta;
        if (ins->sub == 0) { apply_cx(ctx, a, b); apply_1q(ctx, b, cos(th/2.0), cd(0,-sin(th/2.0)), cd(0,-sin(th/2.0)), cos(th/2.0)); apply_cx(ctx, a, b); }
        else if (ins->sub == 1) {
            apply_1q(ctx, a, 1, 0, 0, cd(0,-1)); apply_1q(ctx, b, 1, 0, 0, cd(0,-1));
            apply_cx(ctx, a, b); apply_1q(ctx, b, cos(th/2.0), cd(0,-sin(th/2.0)), cd(0,-sin(th/2.0)), cos(th/2.0)); apply_cx(ctx, a, b);
            apply_1q(ctx, a, 1, 0, 0, cd(0,1)); apply_1q(ctx, b, 1, 0, 0, cd(0,1));
        } else { apply_cx(ctx, a, b); apply_1q(ctx, b, std::exp(cd(0, -th/2.0)), 0, 0, std::exp(cd(0, th/2.0))); apply_cx(ctx, a, b); }
        inject_noise(ctx, a); inject_noise(ctx, b); apply_t1_t2(ctx, a, gate_time_2q); apply_t1_t2(ctx, b, gate_time_2q); break;
    }
    case I_MEASURE: { fuse_flush_all(ctx); int r = measure_qubit(ctx, ins->q0); if (ins->c0 >= 0 && ins->c0 < ctx->n_clbits) ctx->cbits[ins->c0] = r; apply_t1_t2(ctx, ins->q0, gate_time_measure); break; }
    case I_RESET: { fuse_flush_all(ctx); int r = measure_qubit(ctx, ins->q0); if (r == 1) apply_1q(ctx, ins->q0, 0, 1, 1, 0); break; }
    }
}
static void exec_program(QVMContext *ctx, const Program *prog, int verbose) { for (int i = 0; i < prog->count; i++) exec_instr(ctx, &prog->items[i], verbose); }

static double execute_batch_run(Program *prog, int runs) {
    int ncl = 0, nq = 0;
    for (int i = 0; i < prog->count; i++) {
        if (prog->items[i].kind == I_CREG) ncl = prog->items[i].q0;
        if (prog->items[i].kind == I_QREG)  nq  = prog->items[i].q0;
    }
    if (ncl <= 0 || ncl > 24) return -1;

    long long dim = 1LL << nq;
    size_t free_mem = 0, total_mem = 0; cudaMemGetInfo(&free_mem, &total_mem);
    long long max_b = (free_mem - 1000LL*1024*1024) / (dim * sizeof(cdd));
    if (max_b < 1) return -1;
    long long batch_size = max_b > 20000 ? 20000 : max_b; 
    if (batch_size > runs) batch_size = runs;

    cdd *d_states = gpu_batch_alloc(batch_size, dim);
    int *d_cbits = NULL; unsigned long long *d_rngs = NULL;
    cudaMalloc(&d_cbits, batch_size * ncl * sizeof(int)); cudaMalloc(&d_rngs, batch_size * sizeof(unsigned long long));

    unsigned long long *h_rngs = (unsigned long long*)malloc(batch_size * sizeof(unsigned long long));
    Rng temp_rng; rng_seed(&temp_rng, 0xC0FFEEULL);
    for (long long i = 0; i < batch_size; i++) h_rngs[i] = rng_next(&temp_rng);

    int *h_cbits = (int*)malloc(batch_size * ncl * sizeof(int));
    int completed_runs = 0;
    
    g_progress_done = 0; g_progress_total = runs; g_progress_t0 = now_sec(); g_progress_active = 1;

    while (completed_runs < runs) {
        int current_batch = (runs - completed_runs) < batch_size ? (runs - completed_runs) : batch_size;
        gpu_batch_init_zero(d_states, current_batch, dim);
        cudaMemcpy(d_rngs, h_rngs, current_batch * sizeof(unsigned long long), cudaMemcpyHostToDevice);

        for (int i = 0; i < prog->count; i++) {
            Instr *ins = &prog->items[i];
            if (ins->kind == I_GATE1) gpu_batch_apply_1q(d_states, current_batch, dim, ins->q0, ins->m[0][0], ins->m[0][1], ins->m[1][0], ins->m[1][1]);
            else if (ins->kind == I_CX) gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
            else if (ins->kind == I_CZ) gpu_batch_apply_cz(d_states, current_batch, dim, ins->q0, ins->q1);
            else if (ins->kind == I_CP) gpu_batch_apply_cp(d_states, current_batch, dim, ins->q0, ins->q1, ins->theta);
            else if (ins->kind == I_SWAP) gpu_batch_apply_swap(d_states, current_batch, dim, ins->q0, ins->q1);
            else if (ins->kind == I_GATE1_CTRL) gpu_batch_apply_1q_ctrl(d_states, current_batch, dim, ins->q0, ins->q1, ins->m[0][0], ins->m[0][1], ins->m[1][0], ins->m[1][1]);
            else if (ins->kind == I_CCX) gpu_batch_apply_ccx(d_states, current_batch, dim, ins->q0, ins->q1, ins->q2);
            else if (ins->kind == I_CSWAP) {
                gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q2);
                gpu_batch_apply_ccx(d_states, current_batch, dim, ins->q0, ins->q2, ins->q1);
                gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q2);
            } else if (ins->kind == I_GATE2_PAULI) {
                if (ins->sub == 0) {
                    gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                    gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, cos(ins->theta/2.0), cd(0,-sin(ins->theta/2.0)), cd(0,-sin(ins->theta/2.0)), cos(ins->theta/2.0));
                    gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                } else if (ins->sub == 1) {
                    gpu_batch_apply_1q(d_states, current_batch, dim, ins->q0, 1, 0, 0, cd(0,-1));
                    gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, 1, 0, 0, cd(0,-1));
                    gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                    gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, cos(ins->theta/2.0), cd(0,-sin(ins->theta/2.0)), cd(0,-sin(ins->theta/2.0)), cos(ins->theta/2.0));
                    gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                    gpu_batch_apply_1q(d_states, current_batch, dim, ins->q0, 1, 0, 0, cd(0,1));
                    gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, 1, 0, 0, cd(0,1));
                } else {
                    gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                    gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, std::exp(cd(0,-ins->theta/2.0)), 0, 0, std::exp(cd(0,ins->theta/2.0)));
                    gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                }
            } else if (ins->kind == I_MEASURE) gpu_batch_measure(d_states, current_batch, dim, ins->q0, d_cbits, ncl, ins->c0, d_rngs);
        }

        cudaMemcpy(h_cbits, d_cbits, current_batch * ncl * sizeof(int), cudaMemcpyDeviceToHost);
        completed_runs += current_batch; g_progress_done = completed_runs;
    }
    g_progress_active = 0; 

    gpu_batch_free(d_states); cudaFree(d_cbits); cudaFree(d_rngs);
    free(h_cbits); free(h_rngs);
    return 0;
}

// ---------------------------------------------------------
// GUI 线程通信与安全刷新
// ---------------------------------------------------------

typedef struct {
    char* text;
    int is_replace;
} UiLogMsg;

static DWORD WINAPI stress_thread_func(LPVOID param) {
    Program *prog = (Program*)param;
    while (g_stress_active) {
        execute_batch_run(prog, 24);
    }
    return 0;
}

// 安全的追加日志（支持用户上滑查看历史，不会被强制锁底）
void SafeAppendLog(const char* text) {
    int len = GetWindowTextLength(g_hEditLog);
    SCROLLINFO si = {0};
    si.cbSize = sizeof(SCROLLINFO);
    si.fMask = SIF_RANGE | SIF_POS | SIF_PAGE;
    GetScrollInfo(g_hEditLog, SB_VERT, &si);
    
    bool is_at_bottom = (si.nPos + (int)si.nPage >= si.nMax - 1);

    SendMessage(g_hEditLog, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessage(g_hEditLog, EM_REPLACESEL, FALSE, (LPARAM)text);

    if (is_at_bottom) {
        SendMessage(g_hEditLog, EM_SCROLLCARET, 0, 0);
    }
}

// 覆盖最后一行（用于倒计时刷新）
void ReplaceLastLine(const char* newText) {
    int len = GetWindowTextLength(g_hEditLog);
    if (len == 0) return;

    char* buffer = (char*)malloc(len + 1);
    GetWindowText(g_hEditLog, buffer, len + 1);

    int lastLineStart = 0;
    for (int i = len - 1; i >= 0; i--) {
        if (buffer[i] == '\n') {
            lastLineStart = i + 1;
            break;
        }
    }
    free(buffer);

    SendMessage(g_hEditLog, EM_SETSEL, (WPARAM)lastLineStart, (LPARAM)len);
    SendMessage(g_hEditLog, EM_REPLACESEL, FALSE, (LPARAM)newText);
}

// 跑分线程
DWORD WINAPI BenchmarkThread(LPVOID lpParam) {
    nvmlReturn_t nvml_res = nvmlInit();
    nvmlDevice_t nvml_dev;
    unsigned int temp_start = 0, temp_now = 0, temp_end = 0;
    
    if (nvml_res == NVML_SUCCESS) {
        nvmlDeviceGetHandleByIndex(0, &nvml_dev);
        nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &temp_start);
    }

    UiLogMsg* msg = (UiLogMsg*)malloc(sizeof(UiLogMsg));
    msg->text = _strdup("[Test 1] 14 Qubit, 1000 runs...\n");
    msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    Program test14, test24;
    prog_init(&test14);
    prog_push(&test14, {I_QREG, 14}); prog_push(&test14, {I_CREG, 14});
    for (int i=0; i<14; i++) prog_push(&test14, {I_GATE1, i, 0, 0, 0, 0, 0, {{0.707,0.707},{0.707,-0.707}}, 0, 0});
    for (int i=0; i<13; i++) prog_push(&test14, {I_CX, i, i+1});
    for (int i=0; i<14; i++) prog_push(&test14, {I_GATE1, i, 0, 0, 0, 0, 0, {{0,0},{1,0}}, 0, 0});
    for (int i=0; i<14; i++) prog_push(&test14, {I_MEASURE, i, 0, 0, i});

    double t0 = now_sec();
    execute_batch_run(&test14, 1000);
    double t1 = now_sec();
    double time14 = t1 - t0;

    char buf[256];
    snprintf(buf, sizeof(buf), "  -> Done! Time: %.4f s\n\n", time14);
    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg));
    msg->text = _strdup(buf);
    msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    // Test 2
    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg));
    msg->text = _strdup("[Test 2] 24 Qubit, 10 runs...\n");
    msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    prog_init(&test24);
    prog_push(&test24, {I_QREG, 24}); prog_push(&test24, {I_CREG, 24});
    for (int i=0; i<24; i++) prog_push(&test24, {I_GATE1, i, 0, 0, 0, 0, 0, {{0.707,0.707},{0.707,-0.707}}, 0, 0});
    for (int i=0; i<23; i++) prog_push(&test24, {I_CX, i, i+1});
    for (int i=0; i<24; i++) prog_push(&test24, {I_MEASURE, i, 0, 0, i});

    t0 = now_sec();
    execute_batch_run(&test24, 10);
    t1 = now_sec();
    double time24 = t1 - t0;

    snprintf(buf, sizeof(buf), "  -> Done! Time: %.4f s\n\n", time24);
    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg));
    msg->text = _strdup(buf);
    msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    // 算力分
    double score14 = 10000.0 * (0.5 / time14);
    double score24 = 10000.0 * (10.0 / time24);
    double perf_score = score14 * 0.4 + score24 * 0.6;

    // Test 3 烤机
    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg));
    msg->text = _strdup("[Test 3] Thermal Stress Test (5 Minutes)...\n");
    msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    g_stress_active = 1;
    double t_start = now_sec();
    
    HANDLE hStressThread = CreateThread(NULL, 0, stress_thread_func, &test24, 0, NULL);

    while (g_stress_active) {
        double elapsed = now_sec() - t_start;
        int remain = 300 - (int)elapsed;
        if (remain <= 0) { g_stress_active = 0; break; }
        
        if (nvml_res == NVML_SUCCESS) {
            nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &temp_now);
            snprintf(buf, sizeof(buf), "  [Stress Test] Time Remaining: %03d s | Temp: %u C", remain, temp_now);
        } else {
            snprintf(buf, sizeof(buf), "  [Stress Test] Time Remaining: %03d s", remain);
        }
        
        msg = (UiLogMsg*)malloc(sizeof(UiLogMsg));
        msg->text = _strdup(buf);
        msg->is_replace = 1;
        PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);
        
        Sleep(1000);
    }

    if (hStressThread) { WaitForSingleObject(hStressThread, 5000); CloseHandle(hStressThread); }

    if (nvml_res == NVML_SUCCESS) {
        nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &temp_end);
        nvmlShutdown();
    }
    
    int delta = temp_end - temp_start;
    if (delta < 0) delta = 0;
    double temp_score = 200.0 * (50.0 / (delta + 1.0));

    prog_free(&test14);
    prog_free(&test24);

    snprintf(buf, sizeof(buf), "\n========== FINAL RESULT ==========\n"
        "  Performance Score : %.0f\n"
        "  Temperature Score : %.0f\n"
        "  GS QVM MARK 1     : %.0f // %.0f\n"
        "==================================\n", perf_score, temp_score, perf_score, temp_score);
    
    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg));
    msg->text = _strdup(buf);
    msg->is_replace = 0;
    PostMessage(g_hWnd, WM_FINISH_BENCH, 0, (LPARAM)msg);

    return 0;
}

// ---------------------------------------------------------
// 主窗口事件处理
// ---------------------------------------------------------
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CREATE:
            g_hWnd = hwnd;
            CreateWindow("BUTTON", "Start GS QVM MARK 1", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON, 20, 20, 200, 40, hwnd, (HMENU)ID_BTN_START, NULL, NULL);
            g_hEditLog = CreateWindow("EDIT", "", WS_VISIBLE | WS_CHILD | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | WS_VSCROLL, 20, 80, 460, 380, hwnd, (HMENU)ID_EDIT_LOG, NULL, NULL);
            SendMessage(g_hEditLog, WM_SETFONT, (WPARAM)GetStockObject(DEFAULT_GUI_FONT), TRUE);
            break;

        case WM_COMMAND:
            if (LOWORD(wParam) == ID_BTN_START) {
                EnableWindow(GetDlgItem(g_hWnd, ID_BTN_START), FALSE);
                SetWindowText(g_hEditLog, "Starting Benchmark...\n\n");
                CreateThread(NULL, 0, BenchmarkThread, NULL, 0, NULL);
            }
            break;

        case WM_UPDATE_UI: {
            UiLogMsg* msg = (UiLogMsg*)lParam;
            if (msg->is_replace) {
                ReplaceLastLine(msg->text);
            } else {
                SafeAppendLog(msg->text);
            }
            free(msg->text);
            free(msg);
            break;
        }

        case WM_FINISH_BENCH: {
            UiLogMsg* msg = (UiLogMsg*)lParam;
            SafeAppendLog(msg->text);
            free(msg->text);
            free(msg);
            EnableWindow(GetDlgItem(g_hWnd, ID_BTN_START), TRUE);
            break;
        }

        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    if (!gpu_init()) {
        MessageBox(NULL, "GPU Initialization Failed!\nPlease ensure you have an NVIDIA GPU.", "Error", MB_ICONERROR);
        return 1;
    }

    WNDCLASS wc = {0};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "GSQVMWindow";
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(0, "GSQVMWindow", "GS QVM MARK 1 - Quantum Benchmark",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 520, 500, NULL, NULL, hInstance, NULL);

    ShowWindow(hwnd, nCmdShow);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    gpu_shutdown();
    return 0;
}