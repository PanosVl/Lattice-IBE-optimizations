#include "FFT_CUDA_Backend.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cufft.h>

#include <math.h>

namespace {

struct CUDA_State {
    bool initialized;
    cufftHandle forward_plan;
    cufftHandle inverse_plan;
    cufftHandle forward_plan_batch2;
    cufftHandle inverse_plan_batch2;
    long int * d_input_int;
    long int * d_input_int_batch2;
    cufftDoubleComplex * d_freq;
    cufftDoubleComplex * d_freq_batch2;
    cufftDoubleComplex * h_freq;
    cufftDoubleComplex * h_freq_batch2;
};

static CUDA_State g_cuda_state = {false, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static const unsigned int kCudaThreadsPerBlock = 256;

__global__ void KernelIntToComplex(cufftDoubleComplex * out, const long int * in) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N0) {
        out[idx].x = (double)in[idx];
        out[idx].y = 0.0;
    }
}

__global__ void KernelScaleInverse(cufftDoubleComplex * data, double scale) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N0) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

__global__ void KernelScaleInverseN(cufftDoubleComplex * data, double scale, unsigned int n_total) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_total) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

__global__ void KernelIntToComplexN(cufftDoubleComplex * out, const long int * in, unsigned int n_total) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_total) {
        out[idx].x = (double)in[idx];
        out[idx].y = 0.0;
    }
}

int CheckCuda(cudaError_t err) {
    return (err == cudaSuccess) ? 0 : 1;
}

int CheckCufft(cufftResult err) {
    return (err == CUFFT_SUCCESS) ? 0 : 1;
}

unsigned int BlockCountForN0() {
    return (N0 + kCudaThreadsPerBlock - 1) / kCudaThreadsPerBlock;
}

unsigned int BlockCountForLength(unsigned int n_total) {
    return (n_total + kCudaThreadsPerBlock - 1) / kCudaThreadsPerBlock;
}

int CopyIntInputToDevice(const long int * const f) {
    return CheckCuda(cudaMemcpy(g_cuda_state.d_input_int, f, sizeof(long int) * N0, cudaMemcpyHostToDevice));
}

int ConvertDeviceIntsToComplex() {
    KernelIntToComplex<<<BlockCountForN0(), kCudaThreadsPerBlock>>>(g_cuda_state.d_freq, g_cuda_state.d_input_int);
    return CheckCuda(cudaGetLastError());
}

int ExecuteForwardFFT() {
    return CheckCufft(cufftExecZ2Z(g_cuda_state.forward_plan, g_cuda_state.d_freq, g_cuda_state.d_freq, CUFFT_FORWARD));
}

int CopyFreqToHost(CC_t * const f_FFT) {
    if (CheckCuda(cudaMemcpy(g_cuda_state.h_freq, g_cuda_state.d_freq, sizeof(cufftDoubleComplex) * N0, cudaMemcpyDeviceToHost)) != 0) {
        return 1;
    }

    for (unsigned int i = 0; i < N0; ++i) {
        f_FFT[i] = CC_t((RR_t)g_cuda_state.h_freq[i].x, (RR_t)g_cuda_state.h_freq[i].y);
    }

    return 0;
}

} // namespace

int FFT_CUDA_Impl_Init();
void FFT_CUDA_Impl_Cleanup();
int FFT_CUDA_Impl_IsAvailable();
int FFT_CUDA_Impl_IntToFFT(CC_t * f_FFT, const long int * const f);
int FFT_CUDA_Impl_FFTToInt(long int * const f, CC_t const * const f_fft);
int FFT_CUDA_Impl_FFTToReal(double * const f, CC_t const * const f_fft);
int FFT_CUDA_Impl_IntToFFT_Batch(CC_t * f_FFT, const long int * const f, unsigned int batch_count);
int FFT_CUDA_Impl_FFTToInt_Batch(long int * const f, CC_t const * const f_fft, unsigned int batch_count);

namespace {

int EnsureInitialized() {
    if (g_cuda_state.initialized) {
        return 0;
    }
    return FFT_CUDA_Impl_Init();
}

} // namespace

int FFT_CUDA_Impl_IsAvailable() {
    int device_count = 0;
    if (CheckCuda(cudaGetDeviceCount(&device_count)) != 0) {
        return 0;
    }
    return (device_count > 0) ? 1 : 0;
}

int FFT_CUDA_Impl_Init() {
    if (g_cuda_state.initialized) {
        return 0;
    }

    if (!FFT_CUDA_Impl_IsAvailable()) {
        return 1;
    }

    if (CheckCuda(cudaMalloc((void **)&g_cuda_state.d_input_int, sizeof(long int) * N0)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    if (CheckCuda(cudaMalloc((void **)&g_cuda_state.d_freq, sizeof(cufftDoubleComplex) * N0)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    if (CheckCuda(cudaMalloc((void **)&g_cuda_state.d_input_int_batch2, sizeof(long int) * 2 * N0)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    if (CheckCuda(cudaMalloc((void **)&g_cuda_state.d_freq_batch2, sizeof(cufftDoubleComplex) * 2 * N0)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    if (CheckCuda(cudaMallocHost((void **)&g_cuda_state.h_freq, sizeof(cufftDoubleComplex) * N0)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    if (CheckCuda(cudaMallocHost((void **)&g_cuda_state.h_freq_batch2, sizeof(cufftDoubleComplex) * 2 * N0)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    if (CheckCufft(cufftPlan1d(&g_cuda_state.forward_plan, N0, CUFFT_Z2Z, 1)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    if (CheckCufft(cufftPlan1d(&g_cuda_state.inverse_plan, N0, CUFFT_Z2Z, 1)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    int n[1] = {N0};
    if (CheckCufft(cufftPlanMany(&g_cuda_state.forward_plan_batch2, 1, n, 0, 1, N0, 0, 1, N0, CUFFT_Z2Z, 2)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    if (CheckCufft(cufftPlanMany(&g_cuda_state.inverse_plan_batch2, 1, n, 0, 1, N0, 0, 1, N0, CUFFT_Z2Z, 2)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    g_cuda_state.initialized = true;
    return 0;
}

void FFT_CUDA_Impl_Cleanup() {
    if (g_cuda_state.forward_plan != 0) {
        cufftDestroy(g_cuda_state.forward_plan);
        g_cuda_state.forward_plan = 0;
    }

    if (g_cuda_state.inverse_plan != 0) {
        cufftDestroy(g_cuda_state.inverse_plan);
        g_cuda_state.inverse_plan = 0;
    }

    if (g_cuda_state.forward_plan_batch2 != 0) {
        cufftDestroy(g_cuda_state.forward_plan_batch2);
        g_cuda_state.forward_plan_batch2 = 0;
    }

    if (g_cuda_state.inverse_plan_batch2 != 0) {
        cufftDestroy(g_cuda_state.inverse_plan_batch2);
        g_cuda_state.inverse_plan_batch2 = 0;
    }

    if (g_cuda_state.d_input_int != 0) {
        cudaFree(g_cuda_state.d_input_int);
        g_cuda_state.d_input_int = 0;
    }

    if (g_cuda_state.d_input_int_batch2 != 0) {
        cudaFree(g_cuda_state.d_input_int_batch2);
        g_cuda_state.d_input_int_batch2 = 0;
    }

    if (g_cuda_state.d_freq != 0) {
        cudaFree(g_cuda_state.d_freq);
        g_cuda_state.d_freq = 0;
    }

    if (g_cuda_state.d_freq_batch2 != 0) {
        cudaFree(g_cuda_state.d_freq_batch2);
        g_cuda_state.d_freq_batch2 = 0;
    }

    if (g_cuda_state.h_freq != 0) {
        cudaFreeHost(g_cuda_state.h_freq);
        g_cuda_state.h_freq = 0;
    }

    if (g_cuda_state.h_freq_batch2 != 0) {
        cudaFreeHost(g_cuda_state.h_freq_batch2);
        g_cuda_state.h_freq_batch2 = 0;
    }

    g_cuda_state.initialized = false;
}

int FFT_CUDA_Impl_IntToFFT(CC_t * f_FFT, const long int * const f) {
    if (EnsureInitialized() != 0) {
        return 1;
    }

    if (CopyIntInputToDevice(f) != 0) {
        return 1;
    }

    if (ConvertDeviceIntsToComplex() != 0) {
        return 1;
    }

    if (ExecuteForwardFFT() != 0) {
        return 1;
    }

    if (CopyFreqToHost(f_FFT) != 0) {
        return 1;
    }

    return 0;
}

int FFT_CUDA_Impl_FFTToInt(long int * const f, CC_t const * const f_fft) {
    if (EnsureInitialized() != 0) {
        return 1;
    }

    for (unsigned int i = 0; i < N0; ++i) {
        g_cuda_state.h_freq[i].x = (double)real(f_fft[i]);
        g_cuda_state.h_freq[i].y = (double)imag(f_fft[i]);
    }

    if (CheckCuda(cudaMemcpy(g_cuda_state.d_freq, g_cuda_state.h_freq, sizeof(cufftDoubleComplex) * N0, cudaMemcpyHostToDevice)) != 0) {
        return 1;
    }

    if (CheckCufft(cufftExecZ2Z(g_cuda_state.inverse_plan, g_cuda_state.d_freq, g_cuda_state.d_freq, CUFFT_INVERSE)) != 0) {
        return 1;
    }

    double scale = 1.0 / (double)N0;
    KernelScaleInverse<<<BlockCountForN0(), kCudaThreadsPerBlock>>>(g_cuda_state.d_freq, scale);
    if (CheckCuda(cudaGetLastError()) != 0) {
        return 1;
    }

    if (CheckCuda(cudaMemcpy(g_cuda_state.h_freq, g_cuda_state.d_freq, sizeof(cufftDoubleComplex) * N0, cudaMemcpyDeviceToHost)) != 0) {
        return 1;
    }

    for (unsigned int i = 0; i < N0; ++i) {
        f[i] = (long int)llround(g_cuda_state.h_freq[i].x);
    }

    return 0;
}

int FFT_CUDA_Impl_FFTToReal(double * const f, CC_t const * const f_fft) {
    if (EnsureInitialized() != 0) {
        return 1;
    }

    for (unsigned int i = 0; i < N0; ++i) {
        g_cuda_state.h_freq[i].x = (double)real(f_fft[i]);
        g_cuda_state.h_freq[i].y = (double)imag(f_fft[i]);
    }

    if (CheckCuda(cudaMemcpy(g_cuda_state.d_freq, g_cuda_state.h_freq, sizeof(cufftDoubleComplex) * N0, cudaMemcpyHostToDevice)) != 0) {
        return 1;
    }

    if (CheckCufft(cufftExecZ2Z(g_cuda_state.inverse_plan, g_cuda_state.d_freq, g_cuda_state.d_freq, CUFFT_INVERSE)) != 0) {
        return 1;
    }

    double scale = 1.0 / (double)N0;
    KernelScaleInverse<<<BlockCountForN0(), kCudaThreadsPerBlock>>>(g_cuda_state.d_freq, scale);
    if (CheckCuda(cudaGetLastError()) != 0) {
        return 1;
    }

    if (CheckCuda(cudaMemcpy(g_cuda_state.h_freq, g_cuda_state.d_freq, sizeof(cufftDoubleComplex) * N0, cudaMemcpyDeviceToHost)) != 0) {
        return 1;
    }

    for (unsigned int i = 0; i < N0; ++i) {
        f[i] = g_cuda_state.h_freq[i].x;
    }

    return 0;
}

int FFT_CUDA_Impl_IntToFFT_Batch(CC_t * f_FFT, const long int * const f, unsigned int batch_count) {
    if (batch_count == 0) {
        return 0;
    }

    if (batch_count == 1) {
        return FFT_CUDA_Impl_IntToFFT(f_FFT, f);
    }

    if (EnsureInitialized() != 0) {
        return 1;
    }

    if (batch_count == 2) {
        unsigned int total = 2 * N0;
        if (CheckCuda(cudaMemcpy(g_cuda_state.d_input_int_batch2, f, sizeof(long int) * total, cudaMemcpyHostToDevice)) != 0) {
            return 1;
        }

        KernelIntToComplexN<<<BlockCountForLength(total), kCudaThreadsPerBlock>>>(g_cuda_state.d_freq_batch2, g_cuda_state.d_input_int_batch2, total);
        if (CheckCuda(cudaGetLastError()) != 0) {
            return 1;
        }

        if (CheckCufft(cufftExecZ2Z(g_cuda_state.forward_plan_batch2, g_cuda_state.d_freq_batch2, g_cuda_state.d_freq_batch2, CUFFT_FORWARD)) != 0) {
            return 1;
        }

        if (CheckCuda(cudaMemcpy(g_cuda_state.h_freq_batch2, g_cuda_state.d_freq_batch2, sizeof(cufftDoubleComplex) * total, cudaMemcpyDeviceToHost)) != 0) {
            return 1;
        }

        for (unsigned int i = 0; i < total; ++i) {
            f_FFT[i] = CC_t((RR_t)g_cuda_state.h_freq_batch2[i].x, (RR_t)g_cuda_state.h_freq_batch2[i].y);
        }
        return 0;
    }

    for (unsigned int i = 0; i < batch_count; ++i) {
        if (FFT_CUDA_Impl_IntToFFT(f_FFT + i * N0, f + i * N0) != 0) {
            return 1;
        }
    }
    return 0;
}

int FFT_CUDA_Impl_FFTToInt_Batch(long int * const f, CC_t const * const f_fft, unsigned int batch_count) {
    if (batch_count == 0) {
        return 0;
    }

    if (batch_count == 1) {
        return FFT_CUDA_Impl_FFTToInt(f, f_fft);
    }

    if (EnsureInitialized() != 0) {
        return 1;
    }

    if (batch_count == 2) {
        unsigned int total = 2 * N0;
        for (unsigned int i = 0; i < total; ++i) {
            g_cuda_state.h_freq_batch2[i].x = (double)real(f_fft[i]);
            g_cuda_state.h_freq_batch2[i].y = (double)imag(f_fft[i]);
        }

        if (CheckCuda(cudaMemcpy(g_cuda_state.d_freq_batch2, g_cuda_state.h_freq_batch2, sizeof(cufftDoubleComplex) * total, cudaMemcpyHostToDevice)) != 0) {
            return 1;
        }

        if (CheckCufft(cufftExecZ2Z(g_cuda_state.inverse_plan_batch2, g_cuda_state.d_freq_batch2, g_cuda_state.d_freq_batch2, CUFFT_INVERSE)) != 0) {
            return 1;
        }

        double scale = 1.0 / (double)N0;
        KernelScaleInverseN<<<BlockCountForLength(total), kCudaThreadsPerBlock>>>(g_cuda_state.d_freq_batch2, scale, total);
        if (CheckCuda(cudaGetLastError()) != 0) {
            return 1;
        }

        if (CheckCuda(cudaMemcpy(g_cuda_state.h_freq_batch2, g_cuda_state.d_freq_batch2, sizeof(cufftDoubleComplex) * total, cudaMemcpyDeviceToHost)) != 0) {
            return 1;
        }

        for (unsigned int i = 0; i < total; ++i) {
            f[i] = (long int)llround(g_cuda_state.h_freq_batch2[i].x);
        }
        return 0;
    }

    for (unsigned int i = 0; i < batch_count; ++i) {
        if (FFT_CUDA_Impl_FFTToInt(f + i * N0, f_fft + i * N0) != 0) {
            return 1;
        }
    }
    return 0;
}

#endif
