#include "FFT_CUDA_Backend.h"

#ifdef USE_CUDA

int FFT_CUDA_Impl_Init();
void FFT_CUDA_Impl_Cleanup();
int FFT_CUDA_Impl_IsAvailable();
int FFT_CUDA_Impl_IntToFFT(CC_t * f_FFT, const long int * const f);
int FFT_CUDA_Impl_FFTToInt(long int * const f, CC_t const * const f_fft);
int FFT_CUDA_Impl_FFTToReal(double * const f, CC_t const * const f_fft);

int FFT_CUDA_Init() {
    return FFT_CUDA_Impl_Init();
}

void FFT_CUDA_Cleanup() {
    FFT_CUDA_Impl_Cleanup();
}

int FFT_CUDA_IsAvailable() {
    return FFT_CUDA_Impl_IsAvailable();
}

int FFT_CUDA_IntToFFT(CC_t * f_FFT, const long int * const f) {
    return FFT_CUDA_Impl_IntToFFT(f_FFT, f);
}

int FFT_CUDA_FFTToInt(long int * const f, CC_t const * const f_fft) {
    return FFT_CUDA_Impl_FFTToInt(f, f_fft);
}

int FFT_CUDA_FFTToReal(double * const f, CC_t const * const f_fft) {
    return FFT_CUDA_Impl_FFTToReal(f, f_fft);
}

#else

int FFT_CUDA_Init() {
    return 1;
}

void FFT_CUDA_Cleanup() {
}

int FFT_CUDA_IsAvailable() {
    return 0;
}

int FFT_CUDA_IntToFFT(CC_t * f_FFT, const long int * const f) {
    (void)f_FFT;
    (void)f;
    return 1;
}

int FFT_CUDA_FFTToInt(long int * const f, CC_t const * const f_fft) {
    (void)f;
    (void)f_fft;
    return 1;
}

int FFT_CUDA_FFTToReal(double * const f, CC_t const * const f_fft) {
    (void)f;
    (void)f_fft;
    return 1;
}

#endif
