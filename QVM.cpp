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

static volatile LONG g_progress_done = 0, g_progress_total = 0, g_progress_active = 0;
static double g_progress_t0 = 0.0;
static volatile LONG g_stress_active = 0;

HWND g_hWnd;
HWND g_hEditLog;

// ================== 通用批处理执行 (FP64) ==================
static double execute_batch_run_fp64(Program *prog, int runs) {
    int ncl = 0, nq = 0;
    for (int i = 0; i < prog->count; i++) {
        if (prog->items[i].kind == I_CREG) ncl = prog->items[i].q0;
        if (prog->items[i].kind == I_QREG)  nq  = prog->items[i].q0;
    }
    if (ncl <= 0 || ncl > 24) return -1;

    bool has_mid_measure = false;
    for (int i = 0; i < prog->count; i++) {
        if (prog->items[i].kind == I_MEASURE) {
            for (int j = i + 1; j < prog->count; j++) {
                if (prog->items[j].kind != I_MEASURE && prog->items[j].kind != I_CREG && prog->items[j].kind != I_QREG) {
                    has_mid_measure = true; break;
                }
            }
        }
        if (has_mid_measure) break;
    }

    long long dim = 1LL << nq;
    size_t free_mem = 0, total_mem = 0; 
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) return -1;
    
    size_t max_allowed_mem = 3LL * 1024 * 1024 * 1024;
    if (free_mem > max_allowed_mem) free_mem = max_allowed_mem;
    if (free_mem < 500LL * 1024 * 1024) return -1;

    long long batch_size = 1; 
    if (has_mid_measure) {
        long long max_b = (free_mem - 500LL * 1024 * 1024) / (dim * sizeof(cdd));
        if (max_b < 1) max_b = 1;
        batch_size = max_b;
    }
    if (batch_size > runs) batch_size = runs;

    cdd *d_states = gpu_batch_alloc(batch_size, dim);
    int *d_cbits = NULL; unsigned long long *d_rngs = NULL;
    cudaMalloc(&d_cbits, runs * ncl * sizeof(int)); 
    cudaMalloc(&d_rngs, runs * sizeof(unsigned long long));

    unsigned long long *h_rngs = (unsigned long long*)malloc(runs * sizeof(unsigned long long));
    Rng temp_rng; rng_seed(&temp_rng, 0xC0FFEEULL);
    for (int i = 0; i < runs; i++) h_rngs[i] = rng_next(&temp_rng);
    int *h_cbits = (int*)malloc(runs * ncl * sizeof(int));
    
    g_progress_done = 0; g_progress_total = runs; g_progress_t0 = now_sec(); g_progress_active = 1;

    if (!has_mid_measure) {
        gpu_batch_init_zero(d_states, 1, dim);
        cudaMemcpy(d_rngs, h_rngs, runs * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        for (int i = 0; i < prog->count; i++) {
            Instr *ins = &prog->items[i];
            if (ins->kind == I_GATE1) gpu_batch_apply_1q(d_states, 1, dim, ins->q0, ins->m[0][0], ins->m[0][1], ins->m[1][0], ins->m[1][1]);
            else if (ins->kind == I_CX) gpu_batch_apply_cx(d_states, 1, dim, ins->q0, ins->q1);
            else if (ins->kind == I_CZ) gpu_batch_apply_cz(d_states, 1, dim, ins->q0, ins->q1);
            else if (ins->kind == I_CP) gpu_batch_apply_cp(d_states, 1, dim, ins->q0, ins->q1, ins->theta);
            else if (ins->kind == I_SWAP) gpu_batch_apply_swap(d_states, 1, dim, ins->q0, ins->q1);
            else if (ins->kind == I_GATE1_CTRL) gpu_batch_apply_1q_ctrl(d_states, 1, dim, ins->q0, ins->q1, ins->m[0][0], ins->m[0][1], ins->m[1][0], ins->m[1][1]);
            else if (ins->kind == I_CCX) gpu_batch_apply_ccx(d_states, 1, dim, ins->q0, ins->q1, ins->q2);
            else if (ins->kind == I_CSWAP) {
                gpu_batch_apply_cx(d_states, 1, dim, ins->q2, ins->q1); 
                gpu_batch_apply_ccx(d_states, 1, dim, ins->q0, ins->q1, ins->q2); 
                gpu_batch_apply_cx(d_states, 1, dim, ins->q2, ins->q1);
            } else if (ins->kind == I_GATE2_PAULI) {
                if (ins->sub == 0) {
                    gpu_batch_apply_cx(d_states, 1, dim, ins->q0, ins->q1);
                    gpu_batch_apply_1q(d_states, 1, dim, ins->q1, cos(ins->theta/2.0), cd(0,-sin(ins->theta/2.0)), cd(0,-sin(ins->theta/2.0)), cos(ins->theta/2.0));
                    gpu_batch_apply_cx(d_states, 1, dim, ins->q0, ins->q1);
                } else if (ins->sub == 1) {
                    gpu_batch_apply_1q(d_states, 1, dim, ins->q0, cd(0,-1), 0, 0, cd(0,1));
                    gpu_batch_apply_1q(d_states, 1, dim, ins->q1, cd(0,-1), 0, 0, cd(0,1));
                    gpu_batch_apply_cx(d_states, 1, dim, ins->q0, ins->q1);
                    gpu_batch_apply_1q(d_states, 1, dim, ins->q1, cos(ins->theta/2.0), cd(0,-sin(ins->theta/2.0)), cd(0,-sin(ins->theta/2.0)), cos(ins->theta/2.0));
                    gpu_batch_apply_cx(d_states, 1, dim, ins->q0, ins->q1);
                    gpu_batch_apply_1q(d_states, 1, dim, ins->q0, cd(0,1), 0, 0, cd(0,-1));
                    gpu_batch_apply_1q(d_states, 1, dim, ins->q1, cd(0,1), 0, 0, cd(0,-1));
                } else {
                    gpu_batch_apply_cx(d_states, 1, dim, ins->q0, ins->q1);
                    gpu_batch_apply_1q(d_states, 1, dim, ins->q1, std::exp(cd(0,-ins->theta/2.0)), 0, 0, std::exp(cd(0,ins->theta/2.0)));
                    gpu_batch_apply_cx(d_states, 1, dim, ins->q0, ins->q1);
                }
            }
        }
        gpu_sample_from_state(d_states, dim, d_cbits, ncl, d_rngs, runs);
        cudaMemcpy(h_cbits, d_cbits, runs * ncl * sizeof(int), cudaMemcpyDeviceToHost);
        g_progress_done = runs;
    } else {
        int completed_runs = 0;
        while (completed_runs < runs) {
            int current_batch = (runs - completed_runs) < batch_size ? (runs - completed_runs) : batch_size;
            gpu_batch_init_zero(d_states, current_batch, dim);
            cudaMemcpy(d_rngs + completed_runs, h_rngs + completed_runs, current_batch * sizeof(unsigned long long), cudaMemcpyHostToDevice);

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
                    gpu_batch_apply_cx(d_states, current_batch, dim, ins->q2, ins->q1); 
                    gpu_batch_apply_ccx(d_states, current_batch, dim, ins->q0, ins->q1, ins->q2); 
                    gpu_batch_apply_cx(d_states, current_batch, dim, ins->q2, ins->q1);
                } else if (ins->kind == I_GATE2_PAULI) {
                    if (ins->sub == 0) {
                        gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                        gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, cos(ins->theta/2.0), cd(0,-sin(ins->theta/2.0)), cd(0,-sin(ins->theta/2.0)), cos(ins->theta/2.0));
                        gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                    } else if (ins->sub == 1) {
                        gpu_batch_apply_1q(d_states, current_batch, dim, ins->q0, cd(0,-1), 0, 0, cd(0,1));
                        gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, cd(0,-1), 0, 0, cd(0,1));
                        gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                        gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, cos(ins->theta/2.0), cd(0,-sin(ins->theta/2.0)), cd(0,-sin(ins->theta/2.0)), cos(ins->theta/2.0));
                        gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                        gpu_batch_apply_1q(d_states, current_batch, dim, ins->q0, cd(0,1), 0, 0, cd(0,-1));
                        gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, cd(0,1), 0, 0, cd(0,-1));
                    } else {
                        gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                        gpu_batch_apply_1q(d_states, current_batch, dim, ins->q1, std::exp(cd(0,-ins->theta/2.0)), 0, 0, std::exp(cd(0,ins->theta/2.0)));
                        gpu_batch_apply_cx(d_states, current_batch, dim, ins->q0, ins->q1);
                    }
                } else if (ins->kind == I_MEASURE) gpu_batch_measure(d_states, current_batch, dim, ins->q0, d_cbits, ncl, ins->c0, d_rngs + completed_runs);
            }
            cudaMemcpy(h_cbits + completed_runs * ncl, d_cbits, current_batch * ncl * sizeof(int), cudaMemcpyDeviceToHost);
            completed_runs += current_batch; g_progress_done = completed_runs;
        }
    }
    g_progress_active = 0; 

    gpu_batch_free(d_states); cudaFree(d_cbits); cudaFree(d_rngs);
    free(h_cbits); free(h_rngs);
    return 0;
}

// ================== FP32 批处理执行 (加入了强制类型转换) ==================
static double execute_batch_run_fp32(Program *prog, int runs) {
    int ncl = 0, nq = 0;
    for (int i = 0; i < prog->count; i++) {
        if (prog->items[i].kind == I_CREG) ncl = prog->items[i].q0;
        if (prog->items[i].kind == I_QREG)  nq  = prog->items[i].q0;
    }
    if (ncl <= 0 || ncl > 24) return -1;

    long long dim = 1LL << nq;
    size_t free_mem = 0, total_mem = 0; 
    cudaMemGetInfo(&free_mem, &total_mem);
    if (free_mem < 500LL * 1024 * 1024) return -1;

    size_t max_allowed_mem = 3LL * 1024 * 1024 * 1024;
    if (free_mem > max_allowed_mem) free_mem = max_allowed_mem;

    long long batch_size = 1; 
    if (batch_size > runs) batch_size = runs;

    cdf *d_states = gpu_batch_alloc_f32(batch_size, dim);
    int *d_cbits = NULL; unsigned long long *d_rngs = NULL;
    cudaMalloc(&d_cbits, runs * ncl * sizeof(int)); 
    cudaMalloc(&d_rngs, runs * sizeof(unsigned long long));

    unsigned long long *h_rngs = (unsigned long long*)malloc(runs * sizeof(unsigned long long));
    Rng temp_rng; rng_seed(&temp_rng, 0xC0FFEEULL);
    for (int i = 0; i < runs; i++) h_rngs[i] = rng_next(&temp_rng);
    int *h_cbits = (int*)malloc(runs * ncl * sizeof(int));
    
    g_progress_done = 0; g_progress_total = runs; g_progress_t0 = now_sec(); g_progress_active = 1;

    gpu_batch_init_zero_f32(d_states, 1, dim);
    cudaMemcpy(d_rngs, h_rngs, runs * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < prog->count; i++) {
        Instr *ins = &prog->items[i];
        if (ins->kind == I_GATE1) {
            // 强制把 cd(double) 转为 std::complex<float>
            std::complex<float> f00(ins->m[0][0].real(), ins->m[0][0].imag());
            std::complex<float> f01(ins->m[0][1].real(), ins->m[0][1].imag());
            std::complex<float> f10(ins->m[1][0].real(), ins->m[1][0].imag());
            std::complex<float> f11(ins->m[1][1].real(), ins->m[1][1].imag());
            gpu_batch_apply_1q_f32(d_states, 1, dim, ins->q0, f00, f01, f10, f11);
        }
        else if (ins->kind == I_CX) gpu_batch_apply_cx_f32(d_states, 1, dim, ins->q0, ins->q1);
    }
    gpu_sample_from_state_f32(d_states, dim, d_cbits, ncl, d_rngs, runs);
    cudaMemcpy(h_cbits, d_cbits, runs * ncl * sizeof(int), cudaMemcpyDeviceToHost);
    g_progress_done = runs;

    g_progress_active = 0; 
    gpu_batch_free_f32(d_states); cudaFree(d_cbits); cudaFree(d_rngs);
    free(h_cbits); free(h_rngs);
    return 0;
}

static DWORD WINAPI stress_thread_func(LPVOID param) {
    Program *prog = (Program*)param;
    while (g_stress_active) {
        execute_batch_run_fp64(prog, 24); 
    }
    return 0;
}

// --- UI 辅助 ---
typedef struct { char* text; int is_replace; } UiLogMsg;

void SafeAppendLog(const char* text) {
    int len = GetWindowTextLength(g_hEditLog);
    SCROLLINFO si = {0}; si.cbSize = sizeof(SCROLLINFO); si.fMask = SIF_RANGE | SIF_POS | SIF_PAGE;
    GetScrollInfo(g_hEditLog, SB_VERT, &si);
    bool is_at_bottom = (si.nPos + (int)si.nPage >= si.nMax - 1);
    SendMessage(g_hEditLog, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessage(g_hEditLog, EM_REPLACESEL, FALSE, (LPARAM)text);
    if (is_at_bottom) SendMessage(g_hEditLog, EM_SCROLLCARET, 0, 0);
}

void ReplaceLastLine(const char* newText) {
    int len = GetWindowTextLength(g_hEditLog);
    if (len == 0) return;
    char* buffer = (char*)malloc(len + 1); GetWindowText(g_hEditLog, buffer, len + 1);
    int lastLineStart = 0;
    for (int i = len - 1; i >= 0; i--) { if (buffer[i] == '\n') { lastLineStart = i + 1; break; } }
    free(buffer);
    SendMessage(g_hEditLog, EM_SETSEL, (WPARAM)lastLineStart, (LPARAM)len);
    SendMessage(g_hEditLog, EM_REPLACESEL, FALSE, (LPARAM)newText);
}

DWORD WINAPI progress_thread(LPVOID param) {
    (void)param;
    while (g_progress_active) {
        int done = g_progress_done, total = g_progress_total;
        double pct = total > 0 ? (double)done / total : 0.0;
        char buf[256]; snprintf(buf, sizeof(buf), "  [Progress] %.1f%% (%d/%d)", pct * 100.0, done, total);
        UiLogMsg* msg = (UiLogMsg*)malloc(sizeof(UiLogMsg)); msg->text = _strdup(buf); msg->is_replace = 1;
        PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg); Sleep(100);
    }
    return 0;
}

DWORD WINAPI BenchmarkThread(LPVOID lpParam) {
    nvmlReturn_t nvml_res = nvmlInit();
    nvmlDevice_t nvml_dev;
    unsigned int temp_start = 0, temp_now = 0, temp_end = 0;
    if (nvml_res == NVML_SUCCESS) { nvmlDeviceGetHandleByIndex(0, &nvml_dev); nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &temp_start); }

    Program test20;
    prog_init(&test20);
    prog_push(&test20, {I_QREG, 20}); prog_push(&test20, {I_CREG, 20});
    for (int i=0; i<20; i++) prog_push(&test20, {I_GATE1, i, 0, 0, 0, 0, 0, {{0.7071067811865475,0.7071067811865475},{0.7071067811865475,-0.7071067811865475}}, 0, 0});
    for (int i=0; i<19; i++) prog_push(&test20, {I_CX, i, i+1});
    for (int i=0; i<20; i++) prog_push(&test20, {I_GATE1, i, 0, 0, 0, 0, 0, {{0,1},{1,0}}, 0, 0});
    for (int i=0; i<20; i++) prog_push(&test20, {I_MEASURE, i, 0, 0, i});

    HANDLE hProgress = CreateThread(NULL, 0, progress_thread, NULL, 0, NULL);
    char buf[256];

    UiLogMsg* msg = (UiLogMsg*)malloc(sizeof(UiLogMsg));
    msg->text = _strdup("[Test 1] FP64 20 Qubit, 1,000,000 runs...\n"); msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    double t0 = now_sec();
    execute_batch_run_fp64(&test20, 1000000);
    double time64 = now_sec() - t0;
    
    snprintf(buf, sizeof(buf), "  -> Done! Time: %.4f s\n\n", time64);
    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg)); msg->text = _strdup(buf); msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg));
    msg->text = _strdup("[Test 2] FP32 20 Qubit, 1,000,000 runs (Consumer Optimized)...\n"); msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    t0 = now_sec();
    execute_batch_run_fp32(&test20, 1000000);
    double time32 = now_sec() - t0;

    snprintf(buf, sizeof(buf), "  -> Done! Time: %.4f s\n\n", time32);
    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg)); msg->text = _strdup(buf); msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    double score64 = 1000000.0 / time64;
    double score32 = 1000000.0 / time32;
    double perf_score = score64 * 0.45 + score32 * 0.55;

    g_progress_active = 0;
    if (hProgress) { WaitForSingleObject(hProgress, 2000); CloseHandle(hProgress); }

    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg)); msg->text = _strdup("[Test 3] Thermal Stress Test (5 Minutes)...\n"); msg->is_replace = 0;
    PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg);

    Program test24;
    prog_init(&test24);
    prog_push(&test24, {I_QREG, 24}); prog_push(&test24, {I_CREG, 24});
    for (int i=0; i<24; i++) prog_push(&test24, {I_GATE1, i, 0, 0, 0, 0, 0, {{0.7071067811865475,0.7071067811865475},{0.7071067811865475,-0.7071067811865475}}, 0, 0});
    for (int loop=0; loop<10; loop++) {
        for (int i=0; i<23; i++) prog_push(&test24, {I_CX, i, i+1});
        for (int i=0; i<24; i++) prog_push(&test24, {I_GATE1, i, 0, 0, 0, 0, 0, {{0.7071067811865475,0.7071067811865475},{0.7071067811865475,-0.7071067811865475}}, 0, 0});
    }
    for (int i=0; i<24; i++) prog_push(&test24, {I_MEASURE, i, 0, 0, i});

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
        } else { snprintf(buf, sizeof(buf), "  [Stress Test] Time Remaining: %03d s", remain); }
        msg = (UiLogMsg*)malloc(sizeof(UiLogMsg)); msg->text = _strdup(buf); msg->is_replace = 1;
        PostMessage(g_hWnd, WM_UPDATE_UI, 0, (LPARAM)msg); Sleep(1000);
    }
    if (hStressThread) { WaitForSingleObject(hStressThread, 5000); CloseHandle(hStressThread); }

    if (nvml_res == NVML_SUCCESS) { nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &temp_end); nvmlShutdown(); }
    
    int delta = (int)temp_end - (int)temp_start;
    if (delta < 0) delta = -delta;
    if (delta == 0) delta = 1;
    double ratio = (double)delta / 20.0; 
    double temp_score = 1000.0 + 5000.0 * log2(1.0 / ratio); 
    if (temp_score < 0) temp_score = 0;

    prog_free(&test20);
    prog_free(&test24);

    snprintf(buf, sizeof(buf), "\n========== FINAL RESULT ==========\n"
        "  FP64 Time : %.4f s\n"
        "  FP32 Time : %.4f s\n"
        "  Performance Score : %.0f (45%% FP64 + 55%% FP32)\n"
        "  Temperature Score : %.0f\n"
        "  GS QVM MARK 1     : %.0f // %.0f\n"
        "==================================\n", time64, time32, perf_score, temp_score, perf_score, temp_score);
    msg = (UiLogMsg*)malloc(sizeof(UiLogMsg)); msg->text = _strdup(buf); msg->is_replace = 0;
    PostMessage(g_hWnd, WM_FINISH_BENCH, 0, (LPARAM)msg);

    return 0;
}

// ================== Windows 窗口主循环 ==================
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
            if (msg->is_replace) ReplaceLastLine(msg->text);
            else SafeAppendLog(msg->text);
            free(msg->text); free(msg);
            break;
        }
        case WM_FINISH_BENCH: {
            UiLogMsg* msg = (UiLogMsg*)lParam;
            SafeAppendLog(msg->text);
            free(msg->text); free(msg);
            EnableWindow(GetDlgItem(g_hWnd, ID_BTN_START), TRUE);
            break;
        }
        case WM_DESTROY: PostQuitMessage(0); break;
        default: return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    if (!gpu_init()) {
        MessageBox(NULL, "GPU Initialization Failed!\nPlease ensure you have an NVIDIA GPU.", "Error", MB_ICONERROR);
        return 1;
    }
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WindowProc; wc.hInstance = hInstance; wc.lpszClassName = "GSQVMWindow"; wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClass(&wc);
    HWND hwnd = CreateWindowEx(0, "GSQVMWindow", "GS QVM MARK 1 - Quantum Benchmark", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, CW_USEDEFAULT, CW_USEDEFAULT, 520, 500, NULL, NULL, hInstance, NULL);
    ShowWindow(hwnd, nCmdShow);
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) { TranslateMessage(&msg); DispatchMessage(&msg); }
    gpu_shutdown();
    return 0;
}
