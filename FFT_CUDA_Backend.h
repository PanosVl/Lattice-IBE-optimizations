#ifndef LIBE_FFT_CUDA_BACKEND_H
#define LIBE_FFT_CUDA_BACKEND_H

#include "params.h"

// CUDA backend lifecycle.
int FFT_CUDA_Init();
void FFT_CUDA_Cleanup();
int FFT_CUDA_IsAvailable();

// CUDA backend FFT operations.
int FFT_CUDA_IntToFFT(CC_t * f_FFT, const long int * const f);
int FFT_CUDA_FFTToInt(long int * const f, CC_t const * const f_fft);
int FFT_CUDA_FFTToReal(double * const f, CC_t const * const f_fft);

#endif // LIBE_FFT_CUDA_BACKEND_H
