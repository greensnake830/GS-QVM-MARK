#ifndef QVM_CUDA_H
#define QVM_CUDA_H

#include <stdint.h>
#include <complex>
#include <cuda_runtime.h>

typedef double2 cdd;
typedef float2 cdf;

int  gpu_init(void);
void gpu_shutdown(void);
void gpu_sync(void);

// --- FP64 接口 ---
void gpu_apply_1q(cdd *st, long long dim, int q, std::complex<double> m00, std::complex<double> m01, std::complex<double> m10, std::complex<double> m11);
void gpu_apply_1q_ctrl(cdd *st, long long dim, int c, int t, std::complex<double> m00, std::complex<double> m01, std::complex<double> m10, std::complex<double> m11);
void gpu_apply_cx(cdd *st, long long dim, int c, int t);
void gpu_apply_cz(cdd *st, long long dim, int c, int t);
void gpu_apply_cp(cdd *st, long long dim, int c, int t, double theta);
void gpu_apply_swap(cdd *st, long long dim, int a, int b);
void gpu_apply_ccx(cdd *st, long long dim, int c1, int c2, int t);
int  gpu_measure(cdd *st, long long dim, int q, double (*rng_double)(void *), void *rng_ctx);

cdd* gpu_batch_alloc(long long batch_size, long long dim);
void gpu_batch_free(cdd *d_states);
void gpu_batch_init_zero(cdd *d_states, long long batch_size, long long dim);
void gpu_batch_apply_1q(cdd *d_states, long long batch_size, long long dim, int q, std::complex<double> m00, std::complex<double> m01, std::complex<double> m10, std::complex<double> m11);
void gpu_batch_apply_1q_ctrl(cdd *d_states, long long batch_size, long long dim, int c, int t, std::complex<double> m00, std::complex<double> m01, std::complex<double> m10, std::complex<double> m11);
void gpu_batch_apply_cx(cdd *d_states, long long batch_size, long long dim, int c, int t);
void gpu_batch_apply_cz(cdd *d_states, long long batch_size, long long dim, int c, int t);
void gpu_batch_apply_cp(cdd *d_states, long long batch_size, long long dim, int c, int t, double theta);
void gpu_batch_apply_swap(cdd *d_states, long long batch_size, long long dim, int a, int b);
void gpu_batch_apply_ccx(cdd *d_states, long long batch_size, long long dim, int c1, int c2, int t);
void gpu_batch_measure(cdd *d_states, long long batch_size, long long dim, int q, int *d_cbits, int n_clbits, int cbit_idx, unsigned long long *d_rng_states);
void gpu_sample_from_state(cdd *st, long long dim, int *d_cbits, int n_clbits, unsigned long long *d_rng, long long runs);

// --- FP32 接口 (专为消费卡定制) ---
cdf* gpu_batch_alloc_f32(long long batch_size, long long dim);
void gpu_batch_free_f32(cdf *d_states);
void gpu_batch_init_zero_f32(cdf *d_states, long long batch_size, long long dim);
void gpu_batch_apply_1q_f32(cdf *d_states, long long batch_size, long long dim, int q, std::complex<float> m00, std::complex<float> m01, std::complex<float> m10, std::complex<float> m11);
void gpu_batch_apply_cx_f32(cdf *d_states, long long batch_size, long long dim, int c, int t);
void gpu_sample_from_state_f32(cdf *st, long long dim, int *d_cbits, int n_clbits, unsigned long long *d_rng, long long runs);

#endif
