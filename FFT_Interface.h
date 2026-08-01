#ifndef LIBE_FFT_INTERFACE_H
#define LIBE_FFT_INTERFACE_H

#include "params.h"

/**
 * FFT Interface Abstraction Layer
 * 
 * This interface provides a uniform API for FFT operations that can be
 * implemented using different backends (CPU, GPU/cuFFT, etc.).
 * 
 * All functions in this interface maintain the same semantics as the original
 * FFT implementation to ensure drop-in compatibility.
 */

// ============================================================================
// High-level FFT operations for different data types
// ============================================================================

/**
 * Convert integer array to FFT representation
 * @param f_FFT Output: FFT representation (complex array of size N0)
 * @param f Input: Integer array of size N0
 */
void FFT_Interface_IntToFFT(CC_t * f_FFT, const long int * const f);
void FFT_Interface_IntToFFT_Batch(CC_t * f_FFT, const long int * const f, unsigned int batch_count);

/**
 * Convert FFT representation back to integer array
 * @param f Output: Integer array of size N0
 * @param f_fft Input: FFT representation (complex array of size N0)
 */
void FFT_Interface_FFTToInt(long int * const f, CC_t const * const f_fft);
void FFT_Interface_FFTToInt_Batch(long int * const f, CC_t const * const f_fft, unsigned int batch_count);

/**
 * Convert FFT representation to real (double) array
 * @param f Output: Real array of size N0
 * @param f_fft Input: FFT representation (complex array of size N0)
 */
void FFT_Interface_FFTToReal(double * const f, CC_t const * const f_fft);

/**
 * Convert NTL ZZX polynomial to FFT representation
 * @param f_FFT Output: FFT representation (complex array of size N0)
 * @param f Input: NTL polynomial (ZZX)
 */
void FFT_Interface_ZZXToFFT(CC_t * f_FFT, const ZZX f);

/**
 * Convert FFT representation back to NTL ZZX polynomial
 * @param f Output: NTL polynomial (ZZX)
 * @param f_FFT Input: FFT representation (complex array of size N0)
 */
void FFT_Interface_FFTToZZX(ZZX& f, CC_t const * const f_FFT);


// ============================================================================
// Backend management (for future GPU implementation)
// ============================================================================

typedef enum {
    FFT_BACKEND_CPU,
    FFT_BACKEND_GPU
} FFT_Backend_Type;

/**
 * Initialize FFT backend (e.g., allocate GPU memory, create cuFFT plans)
 * @param backend Backend type to use
 * @return 0 on success, non-zero on failure
 */
int FFT_Interface_Init(FFT_Backend_Type backend);

/**
 * Cleanup FFT backend (e.g., free GPU memory, destroy cuFFT plans)
 */
void FFT_Interface_Cleanup();

/**
 * Get current backend type
 */
FFT_Backend_Type FFT_Interface_GetBackend();

/**
 * Check if a backend is available
 */
int FFT_Interface_IsBackendAvailable(FFT_Backend_Type backend);


// ============================================================================
// Performance hints (optional, for optimization)
// ============================================================================

/**
 * Hint that multiple FFT operations will be performed in sequence
 * This can trigger batching optimizations on GPU backends
 */
void FFT_Interface_BeginBatch();

/**
 * End batch processing and synchronize
 */
void FFT_Interface_EndBatch();


#endif // LIBE_FFT_INTERFACE_H
