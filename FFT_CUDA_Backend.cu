#include "FFT_CUDA_Backend.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cufft.h>

#include <math.h>
#include <vector>

namespace {

struct CUDA_State {
    bool initialized;
    cufftHandle forward_plan;
    cufftHandle inverse_plan;
    long int * d_input_int;
    cufftDoubleComplex * d_freq;
};

static CUDA_State g_cuda_state = {false, 0, 0, 0, 0};

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

int CheckCuda(cudaError_t err) {
    return (err == cudaSuccess) ? 0 : 1;
}

int CheckCufft(cufftResult err) {
    return (err == CUFFT_SUCCESS) ? 0 : 1;
}

unsigned int BlockCountForN0() {
    return (N0 + kCudaThreadsPerBlock - 1) / kCudaThreadsPerBlock;
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
    std::vector<cufftDoubleComplex> freq_host(N0);
    if (CheckCuda(cudaMemcpy(&freq_host[0], g_cuda_state.d_freq, sizeof(cufftDoubleComplex) * N0, cudaMemcpyDeviceToHost)) != 0) {
        return 1;
    }

    for (unsigned int i = 0; i < N0; ++i) {
        f_FFT[i] = CC_t((RR_t)freq_host[i].x, (RR_t)freq_host[i].y);
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

    if (CheckCufft(cufftPlan1d(&g_cuda_state.forward_plan, N0, CUFFT_Z2Z, 1)) != 0) {
        FFT_CUDA_Impl_Cleanup();
        return 1;
    }

    if (CheckCufft(cufftPlan1d(&g_cuda_state.inverse_plan, N0, CUFFT_Z2Z, 1)) != 0) {
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

    if (g_cuda_state.d_input_int != 0) {
        cudaFree(g_cuda_state.d_input_int);
        g_cuda_state.d_input_int = 0;
    }

    if (g_cuda_state.d_freq != 0) {
        cudaFree(g_cuda_state.d_freq);
        g_cuda_state.d_freq = 0;
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

    std::vector<cufftDoubleComplex> freq_host(N0);
    for (unsigned int i = 0; i < N0; ++i) {
        freq_host[i].x = (double)real(f_fft[i]);
        freq_host[i].y = (double)imag(f_fft[i]);
    }

    if (CheckCuda(cudaMemcpy(g_cuda_state.d_freq, &freq_host[0], sizeof(cufftDoubleComplex) * N0, cudaMemcpyHostToDevice)) != 0) {
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

    if (CheckCuda(cudaMemcpy(&freq_host[0], g_cuda_state.d_freq, sizeof(cufftDoubleComplex) * N0, cudaMemcpyDeviceToHost)) != 0) {
        return 1;
    }

    for (unsigned int i = 0; i < N0; ++i) {
        f[i] = (long int)llround(freq_host[i].x);
    }

    return 0;
}

int FFT_CUDA_Impl_FFTToReal(double * const f, CC_t const * const f_fft) {
    if (EnsureInitialized() != 0) {
        return 1;
    }

    std::vector<cufftDoubleComplex> freq_host(N0);
    for (unsigned int i = 0; i < N0; ++i) {
        freq_host[i].x = (double)real(f_fft[i]);
        freq_host[i].y = (double)imag(f_fft[i]);
    }

    if (CheckCuda(cudaMemcpy(g_cuda_state.d_freq, &freq_host[0], sizeof(cufftDoubleComplex) * N0, cudaMemcpyHostToDevice)) != 0) {
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

    if (CheckCuda(cudaMemcpy(&freq_host[0], g_cuda_state.d_freq, sizeof(cufftDoubleComplex) * N0, cudaMemcpyDeviceToHost)) != 0) {
        return 1;
    }

    for (unsigned int i = 0; i < N0; ++i) {
        f[i] = freq_host[i].x;
    }

    return 0;
}

#endif
