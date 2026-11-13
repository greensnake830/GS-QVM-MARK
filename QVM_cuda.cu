#include "QVM_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK 256
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while(0)

__device__ __forceinline__ cdd cmul(cdd a, cdd b) { return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ cdd cadd(cdd a, cdd b) { return make_double2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ cdf cmul_f32(cdf a, cdf b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ cdf cadd_f32(cdf a, cdf b) { return make_float2(a.x+b.x, a.y+b.y); }
__device__ inline unsigned long long rng_next_dev(unsigned long long *state) { unsigned long long x = *state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *state = x; return x; }

int gpu_init(void) {
    int dev = 0; if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("[GPU] %s, SM %d.%d, %.1f GB, %d SMs\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0*1024.0*1024.0), prop.multiProcessorCount);
    return 1;
}
void gpu_shutdown(void) { cudaDeviceReset(); }
void gpu_sync(void) { CUDA_CHECK(cudaDeviceSynchronize()); }

// ================== FP64 核心 ==================
cdd* gpu_batch_alloc(long long batch_size, long long dim) { cdd *p = NULL; CUDA_CHECK(cudaMalloc((void**)&p, (size_t)batch_size * dim * sizeof(cdd))); return p; }
void gpu_batch_free(cdd *d_states) { if (d_states) cudaFree(d_states); }

__global__ void kernel_batch_set_zero(cdd *st, long long total) { long long i = blockIdx.x * blockDim.x + threadIdx.x; for (; i < total; i += gridDim.x * blockDim.x) { st[i].x = 0; st[i].y = 0; } }
__global__ void kernel_batch_set_one(cdd *st, long long b_sz, long long dim) { long long b = blockIdx.x * blockDim.x + threadIdx.x; if (b < b_sz) { st[b * dim].x = 1.0; st[b * dim].y = 0.0; } }
void gpu_batch_init_zero(cdd *d_states, long long batch_size, long long dim) {
    long long total = batch_size * dim; long long g = (total + BLOCK - 1) / BLOCK; if (g > 65535) g = 65535;
    kernel_batch_set_zero<<<(int)g, BLOCK>>>(d_states, total);
    g = (batch_size + BLOCK - 1) / BLOCK; if (g > 65535) g = 65535;
    kernel_batch_set_one<<<(int)g, BLOCK>>>(d_states, batch_size, dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_apply_1q(cdd *st, long long b_sz, long long dim, long long step, cdd m00, cdd m01, cdd m10, cdd m11) {
    long long half = dim >> 1; long long total = b_sz * half;
    for (long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += gridDim.x * blockDim.x) {
        long long b = tid / half; long long tid_local = tid % half;
        long long i = (tid_local & (step - 1)) | ((tid_local & ~(step - 1)) << 1); long long j = i | step;
        cdd *l = st + b * dim; cdd x = l[i], y = l[j]; l[i] = cadd(cmul(m00, x), cmul(m01, y)); l[j] = cadd(cmul(m10, x), cmul(m11, y));
    }
}
void gpu_batch_apply_1q(cdd *st, long long b_sz, long long dim, int q, std::complex<double> m00, std::complex<double> m01, std::complex<double> m10, std::complex<double> m11) {
    long long step = 1LL << q; long long half = dim >> 1; long long total = b_sz * half; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    cdd d00 = make_double2(m00.real(), m00.imag()), d01 = make_double2(m01.real(), m01.imag()), d10 = make_double2(m10.real(), m10.imag()), d11 = make_double2(m11.real(), m11.imag());
    kb_apply_1q<<<g, BLOCK>>>(st, b_sz, dim, step, d00, d01, d10, d11); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_apply_1q_ctrl(cdd *st, long long b_sz, long long dim, long long cstep, long long tstep, cdd m00, cdd m01, cdd m10, cdd m11) {
    long long half = dim >> 1; long long total = b_sz * half;
    for (long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += gridDim.x * blockDim.x) {
        long long b = tid / half; long long tid_local = tid % half;
        long long i = (tid_local & (tstep - 1)) | ((tid_local & ~(tstep - 1)) << 1);
        if ((i & tstep) || !(i & cstep)) continue;
        long long j = i | tstep; if (j >= dim) continue;
        cdd *l = st + b * dim; cdd x = l[i], y = l[j]; l[i] = cadd(cmul(m00, x), cmul(m01, y)); l[j] = cadd(cmul(m10, x), cmul(m11, y));
    }
}
void gpu_batch_apply_1q_ctrl(cdd *st, long long b_sz, long long dim, int c, int t, std::complex<double> m00, std::complex<double> m01, std::complex<double> m10, std::complex<double> m11) {
    long long cstep = 1LL << c, tstep = 1LL << t; long long half = dim >> 1; long long total = b_sz * half; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    cdd d00 = make_double2(m00.real(), m00.imag()), d01 = make_double2(m01.real(), m01.imag()), d10 = make_double2(m10.real(), m10.imag()), d11 = make_double2(m11.real(), m11.imag());
    kb_apply_1q_ctrl<<<g, BLOCK>>>(st, b_sz, dim, cstep, tstep, d00, d01, d10, d11); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_cx(cdd *st, long long b_sz, long long dim, long long cstep, long long tstep) {
    long long half = dim >> 1; long long total = b_sz * half;
    for (long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += gridDim.x * blockDim.x) {
        long long b = tid / half; long long tid_local = tid % half;
        long long i = (tid_local & (tstep - 1)) | ((tid_local & ~(tstep - 1)) << 1);
        if ((i & tstep) || !(i & cstep)) continue;
        long long j = i | tstep; if (j >= dim) continue;
        cdd *l = st + b * dim; cdd tmp = l[i]; l[i] = l[j]; l[j] = tmp;
    }
}
void gpu_batch_apply_cx(cdd *st, long long b_sz, long long dim, int c, int t) {
    long long cstep = 1LL << c, tstep = 1LL << t; long long half = dim >> 1; long long total = b_sz * half; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    kb_cx<<<g, BLOCK>>>(st, b_sz, dim, cstep, tstep); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_cz(cdd *st, long long b_sz, long long dim, long long cstep, long long tstep) {
    long long total = b_sz * dim;
    for (long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += gridDim.x * blockDim.x) {
        long long i = tid % dim; if ((i & cstep) && (i & tstep)) { st[tid].x = -st[tid].x; st[tid].y = -st[tid].y; }
    }
}
void gpu_batch_apply_cz(cdd *st, long long b_sz, long long dim, int c, int t) {
    long long cstep = 1LL << c, tstep = 1LL << t; long long total = b_sz * dim; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    kb_cz<<<g, BLOCK>>>(st, b_sz, dim, cstep, tstep); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_cp(cdd *st, long long b_sz, long long dim, long long cstep, long long tstep, double cr, double ci) {
    long long total = b_sz * dim;
    for (long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += gridDim.x * blockDim.x) {
        long long i = tid % dim; if ((i & cstep) && (i & tstep)) { cdd v = st[tid]; st[tid] = make_double2(v.x*cr - v.y*ci, v.x*ci + v.y*cr); }
    }
}
void gpu_batch_apply_cp(cdd *st, long long b_sz, long long dim, int c, int t, double theta) {
    long long cstep = 1LL << c, tstep = 1LL << t; long long total = b_sz * dim; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    kb_cp<<<g, BLOCK>>>(st, b_sz, dim, cstep, tstep, cos(theta), sin(theta)); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_swap(cdd *st, long long b_sz, long long dim, long long astep, long long bstep) {
    long long total = b_sz * dim;
    for (long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += gridDim.x * blockDim.x) {
        long long b = tid / dim; long long i = tid % dim;
        if ((i & astep) || !(i & bstep)) continue;
        long long j = i ^ astep ^ bstep;
        cdd *l = st + b * dim; cdd tmp = l[i]; l[i] = l[j]; l[j] = tmp;
    }
}
void gpu_batch_apply_swap(cdd *st, long long b_sz, long long dim, int a, int b) {
    if (a == b) return;
    long long astep = 1LL << a, bstep = 1LL << b; long long total = b_sz * dim; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    kb_swap<<<g, BLOCK>>>(st, b_sz, dim, astep, bstep); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_ccx(cdd *st, long long b_sz, long long dim, long long c1s, long long c2s, long long ts) {
    long long half = dim >> 1; long long total = b_sz * half;
    for (long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += gridDim.x * blockDim.x) {
        long long b = tid / half; long long tid_local = tid % half;
        long long i = (tid_local & (ts - 1)) | ((tid_local & ~(ts - 1)) << 1);
        if ((i & ts) || !(i & c1s) || !(i & c2s)) continue;
        long long j = i | ts; if (j >= dim) continue;
        cdd *l = st + b * dim; cdd tmp = l[i]; l[i] = l[j]; l[j] = tmp;
    }
}
void gpu_batch_apply_ccx(cdd *st, long long b_sz, long long dim, int c1, int c2, int t) {
    long long c1s = 1LL << c1, c2s = 1LL << c2, ts = 1LL << t; long long half = dim >> 1; long long total = b_sz * half; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    kb_ccx<<<g, BLOCK>>>(st, b_sz, dim, c1s, c2s, ts); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_measure_multiblock(cdd *st, long long b_sz, long long dim, long long step, double *global_p1, int *d_cbits, int n_clbits, int cbit_idx, unsigned long long *d_rng) {
    long long b = blockIdx.y; if (b >= b_sz) return;
    cdd *l = st + b * dim; __shared__ double sdata[BLOCK]; double sum = 0.0;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += blockDim.x * gridDim.x)
        if (i & step) sum += l[i].x * l[i].x + l[i].y * l[i].y;
    sdata[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(&global_p1[b], sdata[0]);
    __threadfence(); 
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double p1 = global_p1[b]; if (p1 < 0.0) p1 = 0.0; if (p1 > 1.0) p1 = 1.0;
        double r = (rng_next_dev(&d_rng[b]) >> 11) * (1.0 / 9007199254740992.0);
        int res = (r < p1) ? 1 : 0; double norm = res ? p1 : (1.0 - p1);
        if (norm < 1e-12) { res = 1 - res; norm = res ? p1 : (1.0 - p1); }
        if (norm < 1e-12) norm = 1e-12; global_p1[b] = 1.0 / sqrt(norm); d_cbits[b * n_clbits + cbit_idx] = res;
    }
    __threadfence(); __syncthreads();
    int res = d_cbits[b * n_clbits + cbit_idx]; double inv = global_p1[b];
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += blockDim.x * gridDim.x) {
        if (((i & step) ? 1 : 0) == res) { l[i].x *= inv; l[i].y *= inv; } else { l[i].x = 0; l[i].y = 0; }
    }
}
void gpu_batch_measure(cdd *st, long long b_sz, long long dim, int q, int *d_cbits, int n_clbits, int cbit_idx, unsigned long long *d_rng) {
    long long step = 1LL << q; double *d_p1 = NULL; 
    cudaMalloc(&d_p1, b_sz * sizeof(double)); cudaMemset(d_p1, 0, b_sz * sizeof(double));
    dim3 grid(170, (unsigned int)b_sz); 
    kb_measure_multiblock<<<grid, BLOCK>>>(st, b_sz, dim, step, d_p1, d_cbits, n_clbits, cbit_idx, d_rng);
    cudaFree(d_p1); CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sample_cdf(cdd *st, long long dim, int *d_cbits, int n_clbits, unsigned long long *d_rng, long long b_sz) {
    long long b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= b_sz) return;
    double r = (rng_next_dev(&d_rng[b]) >> 11) * (1.0 / 9007199254740992.0);
    double cdf = 0.0; int result = dim - 1;
    for (long long i = 0; i < dim; i++) { cdf += st[i].x * st[i].x + st[i].y * st[i].y; if (r < cdf) { result = (int)i; break; } }
    for (int c = 0; c < n_clbits; c++) d_cbits[b * n_clbits + c] = (result >> c) & 1;
}
void gpu_sample_from_state(cdd *st, long long dim, int *d_cbits, int n_clbits, unsigned long long *d_rng, long long runs) {
    int g = (int)((runs + BLOCK - 1) / BLOCK);
    k_sample_cdf<<<g, BLOCK>>>(st, dim, d_cbits, n_clbits, d_rng, runs); CUDA_CHECK(cudaGetLastError());
}

// ================== FP32 专供内核 ==================
cdf* gpu_batch_alloc_f32(long long batch_size, long long dim) { cdf *p = NULL; CUDA_CHECK(cudaMalloc((void**)&p, (size_t)batch_size * dim * sizeof(cdf))); return p; }
void gpu_batch_free_f32(cdf *d_states) { if (d_states) cudaFree(d_states); }

__global__ void kernel_batch_set_zero_f32(cdf *st, long long total) { long long i = blockIdx.x * blockDim.x + threadIdx.x; for (; i < total; i += gridDim.x * blockDim.x) { st[i].x = 0; st[i].y = 0; } }
__global__ void kernel_batch_set_one_f32(cdf *st, long long b_sz, long long dim) { long long b = blockIdx.x * blockDim.x + threadIdx.x; if (b < b_sz) { st[b * dim].x = 1.0f; st[b * dim].y = 0.0f; } }
void gpu_batch_init_zero_f32(cdf *d_states, long long batch_size, long long dim) {
    long long total = batch_size * dim; long long g = (total + BLOCK - 1) / BLOCK; if (g > 65535) g = 65535;
    kernel_batch_set_zero_f32<<<(int)g, BLOCK>>>(d_states, total);
    g = (batch_size + BLOCK - 1) / BLOCK; if (g > 65535) g = 65535;
    kernel_batch_set_one_f32<<<(int)g, BLOCK>>>(d_states, batch_size, dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_apply_1q_f32(cdf *st, long long b_sz, long long dim, long long step, cdf m00, cdf m01, cdf m10, cdf m11) {
    long long half = dim >> 1; long long total = b_sz * half;
    for (long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += gridDim.x * blockDim.x) {
        long long b = tid / half; long long tid_local = tid % half;
        long long i = (tid_local & (step - 1)) | ((tid_local & ~(step - 1)) << 1); long long j = i | step;
        cdf *l = st + b * dim; cdf x = l[i], y = l[j]; l[i] = cadd_f32(cmul_f32(m00, x), cmul_f32(m01, y)); l[j] = cadd_f32(cmul_f32(m10, x), cmul_f32(m11, y));
    }
}
void gpu_batch_apply_1q_f32(cdf *st, long long b_sz, long long dim, int q, std::complex<float> m00, std::complex<float> m01, std::complex<float> m10, std::complex<float> m11) {
    long long step = 1LL << q; long long half = dim >> 1; long long total = b_sz * half; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    cdf d00 = make_float2(m00.real(), m00.imag()), d01 = make_float2(m01.real(), m01.imag()), d10 = make_float2(m10.real(), m10.imag()), d11 = make_float2(m11.real(), m11.imag());
    kb_apply_1q_f32<<<g, BLOCK>>>(st, b_sz, dim, step, d00, d01, d10, d11); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_cx_f32(cdf *st, long long b_sz, long long dim, long long cstep, long long tstep) {
    long long half = dim >> 1; long long total = b_sz * half;
    for (long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += gridDim.x * blockDim.x) {
        long long b = tid / half; long long tid_local = tid % half;
        long long i = (tid_local & (tstep - 1)) | ((tid_local & ~(tstep - 1)) << 1);
        if ((i & tstep) || !(i & cstep)) continue;
        long long j = i | tstep; if (j >= dim) continue;
        cdf *l = st + b * dim; cdf tmp = l[i]; l[i] = l[j]; l[j] = tmp;
    }
}
void gpu_batch_apply_cx_f32(cdf *st, long long b_sz, long long dim, int c, int t) {
    long long cstep = 1LL << c, tstep = 1LL << t; long long half = dim >> 1; long long total = b_sz * half; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    kb_cx_f32<<<g, BLOCK>>>(st, b_sz, dim, cstep, tstep); CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sample_cdf_f32(cdf *st, long long dim, int *d_cbits, int n_clbits, unsigned long long *d_rng, long long b_sz) {
    long long b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= b_sz) return;
    double r = (rng_next_dev(&d_rng[b]) >> 11) * (1.0 / 9007199254740992.0);
    double cdf = 0.0; int result = dim - 1;
    for (long long i = 0; i < dim; i++) { cdf += st[i].x * st[i].x + st[i].y * st[i].y; if (r < cdf) { result = (int)i; break; } }
    for (int c = 0; c < n_clbits; c++) d_cbits[b * n_clbits + c] = (result >> c) & 1;
}
void gpu_sample_from_state_f32(cdf *st, long long dim, int *d_cbits, int n_clbits, unsigned long long *d_rng, long long runs) {
    int g = (int)((runs + BLOCK - 1) / BLOCK);
    k_sample_cdf_f32<<<g, BLOCK>>>(st, dim, d_cbits, n_clbits, d_rng, runs); CUDA_CHECK(cudaGetLastError());
}
