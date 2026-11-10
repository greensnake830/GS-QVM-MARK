#include "QVM_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK 256
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while(0)

__device__ __forceinline__ cdd cmul(cdd a, cdd b) { return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ cdd cadd(cdd a, cdd b) { return make_double2(a.x+b.x, a.y+b.y); }
__device__ inline unsigned long long rng_next_dev(unsigned long long *state) { unsigned long long x = *state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *state = x; return x; }

int gpu_init(void) {
    int dev = 0; if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("[GPU] %s, SM %d.%d, %.1f GB, %d SMs\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0*1024.0*1024.0), prop.multiProcessorCount);
    return 1;
}
void gpu_shutdown(void) { cudaDeviceReset(); }
void gpu_sync(void) { CUDA_CHECK(cudaDeviceSynchronize()); }

cdd* gpu_batch_alloc(long long batch_size, long long dim) { cdd *p = NULL; CUDA_CHECK(cudaMalloc((void**)&p, (size_t)batch_size * dim * sizeof(cdd))); return p; }
void gpu_batch_free(cdd *d_states) { if (d_states) cudaFree(d_states); }

__global__ void kernel_batch_set_zero(cdd *st, long long total_elems) { long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; for (; i < total_elems; i += (long long)gridDim.x * blockDim.x) { st[i].x = 0; st[i].y = 0; } }
__global__ void kernel_batch_set_one(cdd *st, long long batch_size, long long dim) { long long b = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (b < batch_size) { st[b * dim].x = 1.0; st[b * dim].y = 0.0; } }
void gpu_batch_init_zero(cdd *d_states, long long batch_size, long long dim) {
    long long total = batch_size * dim; long long g = (total + BLOCK - 1) / BLOCK; if (g > 65535) g = 65535;
    kernel_batch_set_zero<<<(int)g, BLOCK>>>(d_states, total);
    g = (batch_size + BLOCK - 1) / BLOCK; if (g > 65535) g = 65535;
    kernel_batch_set_one<<<(int)g, BLOCK>>>(d_states, batch_size, dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_apply_1q(cdd *st, long long dim, long long step, cdd m00, cdd m01, cdd m10, cdd m11) {
    long long half = dim >> 1;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < half; tid += (long long)gridDim.x * blockDim.x) {
        long long i = (tid & (step - 1)) | ((tid & ~(step - 1)) << 1); long long j = i | step;
        cdd x = st[i], y = st[j]; st[i] = cadd(cmul(m00, x), cmul(m01, y)); st[j] = cadd(cmul(m10, x), cmul(m11, y));
    }
}
void gpu_apply_1q(cdd *st, long long dim, int q, std::complex<double> m00, std::complex<double> m01, std::complex<double> m10, std::complex<double> m11) {
    long long step = 1LL << q; long long half = dim >> 1; int g = (int)((half + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    cdd d00 = make_double2(m00.real(), m00.imag()), d01 = make_double2(m01.real(), m01.imag()), d10 = make_double2(m10.real(), m10.imag()), d11 = make_double2(m11.real(), m11.imag());
    k_apply_1q<<<g, BLOCK>>>(st, dim, step, d00, d01, d10, d11); CUDA_CHECK(cudaGetLastError());
}

__global__ void k_apply_1q_ctrl(cdd *st, long long dim, long long cstep, long long tstep, cdd m00, cdd m01, cdd m10, cdd m11) {
    long long half = dim >> 1;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < half; tid += (long long)gridDim.x * blockDim.x) {
        long long i = (tid & (tstep - 1)) | ((tid & ~(tstep - 1)) << 1);
        if ((i & tstep) || !(i & cstep)) continue;
        long long j = i | tstep; if (j >= dim) continue;
        cdd x = st[i], y = st[j]; st[i] = cadd(cmul(m00, x), cmul(m01, y)); st[j] = cadd(cmul(m10, x), cmul(m11, y));
    }
}
void gpu_apply_1q_ctrl(cdd *st, long long dim, int c, int t, std::complex<double> m00, std::complex<double> m01, std::complex<double> m10, std::complex<double> m11) {
    long long cstep = 1LL << c, tstep = 1LL << t; long long half = dim >> 1; int g = (int)((half + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    cdd d00 = make_double2(m00.real(), m00.imag()), d01 = make_double2(m01.real(), m01.imag()), d10 = make_double2(m10.real(), m10.imag()), d11 = make_double2(m11.real(), m11.imag());
    k_apply_1q_ctrl<<<g, BLOCK>>>(st, dim, cstep, tstep, d00, d01, d10, d11); CUDA_CHECK(cudaGetLastError());
}

__global__ void k_cx(cdd *st, long long dim, long long cstep, long long tstep) {
    long long half = dim >> 1;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < half; tid += (long long)gridDim.x * blockDim.x) {
        long long i = (tid & (tstep - 1)) | ((tid & ~(tstep - 1)) << 1);
        if ((i & tstep) || !(i & cstep)) continue;
        long long j = i | tstep; if (j >= dim) continue;
        cdd tmp = st[i]; st[i] = st[j]; st[j] = tmp;
    }
}
void gpu_apply_cx(cdd *st, long long dim, int c, int t) {
    long long cstep = 1LL << c, tstep = 1LL << t; long long half = dim >> 1; int g = (int)((half + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    k_cx<<<g, BLOCK>>>(st, dim, cstep, tstep); CUDA_CHECK(cudaGetLastError());
}

__global__ void k_cz(cdd *st, long long dim, long long cstep, long long tstep) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < dim; i += (long long)gridDim.x * blockDim.x)
        if ((i & cstep) && (i & tstep)) { st[i].x = -st[i].x; st[i].y = -st[i].y; }
}
void gpu_apply_cz(cdd *st, long long dim, int c, int t) {
    long long cstep = 1LL << c, tstep = 1LL << t; int g = (int)((dim + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    k_cz<<<g, BLOCK>>>(st, dim, cstep, tstep); CUDA_CHECK(cudaGetLastError());
}

__global__ void k_cp(cdd *st, long long dim, long long cstep, long long tstep, double cr, double ci) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < dim; i += (long long)gridDim.x * blockDim.x)
        if ((i & cstep) && (i & tstep)) { cdd v = st[i]; st[i] = make_double2(v.x*cr - v.y*ci, v.x*ci + v.y*cr); }
}
void gpu_apply_cp(cdd *st, long long dim, int c, int t, double theta) {
    long long cstep = 1LL << c, tstep = 1LL << t; int g = (int)((dim + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    k_cp<<<g, BLOCK>>>(st, dim, cstep, tstep, cos(theta), sin(theta)); CUDA_CHECK(cudaGetLastError());
}

__global__ void k_swap(cdd *st, long long dim, long long astep, long long bstep) {
    long long half = dim >> 1;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < half; tid += (long long)gridDim.x * blockDim.x) {
        long long i = (tid & (astep - 1)) | ((tid & ~(astep - 1)) << 1);
        if (i & bstep) continue;
        long long j = i | astep; if (j >= dim) continue;
        cdd tmp = st[i]; st[i] = st[j]; st[j] = tmp;
    }
}
void gpu_apply_swap(cdd *st, long long dim, int a, int b) {
    if (a == b) return;
    long long astep = 1LL << a, bstep = 1LL << b; long long half = dim >> 1; int g = (int)((half + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    k_swap<<<g, BLOCK>>>(st, dim, astep, bstep); CUDA_CHECK(cudaGetLastError());
}

__global__ void k_ccx(cdd *st, long long dim, long long c1s, long long c2s, long long ts) {
    long long half = dim >> 1;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < half; tid += (long long)gridDim.x * blockDim.x) {
        long long i = (tid & (ts - 1)) | ((tid & ~(ts - 1)) << 1);
        if ((i & ts) || !(i & c1s) || !(i & c2s)) continue;
        long long j = i | ts; if (j >= dim) continue;
        cdd tmp = st[i]; st[i] = st[j]; st[j] = tmp;
    }
}
void gpu_apply_ccx(cdd *st, long long dim, int c1, int c2, int t) {
    long long c1s = 1LL << c1, c2s = 1LL << c2, ts = 1LL << t; long long half = dim >> 1; int g = (int)((half + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    k_ccx<<<g, BLOCK>>>(st, dim, c1s, c2s, ts); CUDA_CHECK(cudaGetLastError());
}

__global__ void k_prob1(const cdd *st, long long dim, long long step, double *out_p1) {
    __shared__ double sdata[BLOCK]; double sum = 0.0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < dim; i += (long long)gridDim.x * blockDim.x)
        if (i & step) sum += st[i].x*st[i].x + st[i].y*st[i].y;
    sdata[threadIdx.x] = sum; __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out_p1, sdata[0]);
}
__global__ void k_project(cdd *st, long long dim, long long step, int result, double inv) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < dim; i += (long long)gridDim.x * blockDim.x) {
        if (((i & step) ? 1 : 0) == result) { st[i].x *= inv; st[i].y *= inv; } else { st[i].x = 0; st[i].y = 0; }
    }
}
int gpu_measure(cdd *st, long long dim, int q, double (*rng_double)(void *), void *rng_ctx) {
    long long step = 1LL << q; double *d_p1 = NULL; CUDA_CHECK(cudaMalloc((void**)&d_p1, sizeof(double))); CUDA_CHECK(cudaMemset(d_p1, 0, sizeof(double)));
    int g = (int)((dim + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    k_prob1<<<g, BLOCK>>>(st, dim, step, d_p1);
    double p1 = 0.0; CUDA_CHECK(cudaMemcpy(&p1, d_p1, sizeof(double), cudaMemcpyDeviceToHost)); cudaFree(d_p1);
    if (p1 < 0.0) p1 = 0.0; if (p1 > 1.0) p1 = 1.0;
    int result = (rng_double(rng_ctx) < p1) ? 1 : 0; double norm = result ? p1 : (1.0 - p1);
    if (norm < 1e-12) { result = 1 - result; norm = result ? p1 : (1.0 - p1); }
    if (norm < 1e-12) norm = 1e-12; double inv = 1.0 / sqrt(norm);
    int gp = (int)((dim + BLOCK - 1) / BLOCK); if (gp > 65535) gp = 65535;
    k_project<<<gp, BLOCK>>>(st, dim, step, result, inv); CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

__global__ void kb_apply_1q(cdd *st, long long b_sz, long long dim, long long step, cdd m00, cdd m01, cdd m10, cdd m11) {
    long long half = dim >> 1; long long total = b_sz * half;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < total; tid += (long long)gridDim.x * blockDim.x) {
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
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < total; tid += (long long)gridDim.x * blockDim.x) {
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
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < total; tid += (long long)gridDim.x * blockDim.x) {
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
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < total; tid += (long long)gridDim.x * blockDim.x) {
        long long i = tid % dim; if ((i & cstep) && (i & tstep)) { st[tid].x = -st[tid].x; st[tid].y = -st[tid].y; }
    }
}
void gpu_batch_apply_cz(cdd *st, long long b_sz, long long dim, int c, int t) {
    long long cstep = 1LL << c, tstep = 1LL << t; long long total = b_sz * dim; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    kb_cz<<<g, BLOCK>>>(st, b_sz, dim, cstep, tstep); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_cp(cdd *st, long long b_sz, long long dim, long long cstep, long long tstep, double cr, double ci) {
    long long total = b_sz * dim;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < total; tid += (long long)gridDim.x * blockDim.x) {
        long long i = tid % dim; if ((i & cstep) && (i & tstep)) { cdd v = st[tid]; st[tid] = make_double2(v.x*cr - v.y*ci, v.x*ci + v.y*cr); }
    }
}
void gpu_batch_apply_cp(cdd *st, long long b_sz, long long dim, int c, int t, double theta) {
    long long cstep = 1LL << c, tstep = 1LL << t; long long total = b_sz * dim; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    kb_cp<<<g, BLOCK>>>(st, b_sz, dim, cstep, tstep, cos(theta), sin(theta)); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_swap(cdd *st, long long b_sz, long long dim, long long astep, long long bstep) {
    long long half = dim >> 1; long long total = b_sz * half;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < total; tid += (long long)gridDim.x * blockDim.x) {
        long long b = tid / half; long long tid_local = tid % half;
        long long i = (tid_local & (astep - 1)) | ((tid_local & ~(astep - 1)) << 1);
        if (i & bstep) continue;
        long long j = i | astep; if (j >= dim) continue;
        cdd *l = st + b * dim; cdd tmp = l[i]; l[i] = l[j]; l[j] = tmp;
    }
}
void gpu_batch_apply_swap(cdd *st, long long b_sz, long long dim, int a, int b) {
    if (a == b) return;
    long long astep = 1LL << a, bstep = 1LL << b; long long half = dim >> 1; long long total = b_sz * half; int g = (int)((total + BLOCK - 1) / BLOCK); if (g > 65535) g = 65535;
    kb_swap<<<g, BLOCK>>>(st, b_sz, dim, astep, bstep); CUDA_CHECK(cudaGetLastError());
}

__global__ void kb_ccx(cdd *st, long long b_sz, long long dim, long long c1s, long long c2s, long long ts) {
    long long half = dim >> 1; long long total = b_sz * half;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x; tid < total; tid += (long long)gridDim.x * blockDim.x) {
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

__global__ void kb_measure(cdd *st, long long b_sz, long long dim, long long step, int *d_cbits, int n_clbits, int cbit_idx, unsigned long long *d_rng) {
    long long b = blockIdx.x;
    if (b >= b_sz) return;
    cdd *l = st + b * dim;
    __shared__ double sdata[BLOCK];
    double sum = 0.0;
    for (long long i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i & step) sum += l[i].x*l[i].x + l[i].y*l[i].y;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ int res_shared;
    __shared__ double inv_shared;
    if (threadIdx.x == 0) {
        double p1 = sdata[0];
        if (p1 < 0.0) p1 = 0.0; if (p1 > 1.0) p1 = 1.0;
        double r = (rng_next_dev(&d_rng[b]) >> 11) * (1.0 / 9007199254740992.0);
        int res = (r < p1) ? 1 : 0; double norm = res ? p1 : (1.0 - p1);
        if (norm < 1e-12) { res = 1 - res; norm = res ? p1 : (1.0 - p1); }
        if (norm < 1e-12) norm = 1e-12; 
        res_shared = res;
        inv_shared = 1.0 / sqrt(norm);
        d_cbits[b * n_clbits + cbit_idx] = res;
    }
    __syncthreads();
    int res = res_shared;
    double inv = inv_shared;
    for (long long i = threadIdx.x; i < dim; i += blockDim.x) {
        if (((i & step) ? 1 : 0) == res) { l[i].x *= inv; l[i].y *= inv; } else { l[i].x = 0; l[i].y = 0; }
    }
}
void gpu_batch_measure(cdd *st, long long b_sz, long long dim, int q, int *d_cbits, int n_clbits, int cbit_idx, unsigned long long *d_rng) {
    long long step = 1LL << q; int g = (int)b_sz; if (g > 65535) g = 65535;
    kb_measure<<<g, BLOCK>>>(st, b_sz, dim, step, d_cbits, n_clbits, cbit_idx, d_rng); CUDA_CHECK(cudaGetLastError());
}