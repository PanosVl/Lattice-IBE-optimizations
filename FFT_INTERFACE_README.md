# FFT Interface Abstraction Layer

## Overview

This document describes the FFT abstraction layer created to enable future GPU acceleration using cuFFT. The interface provides a uniform API that abstracts the underlying FFT implementation, allowing seamless switching between CPU and GPU backends.

## Architecture

### Files

- **FFT_Interface.h** - Interface declarations and API
- **FFT_Interface.cc** - Current CPU-based implementation
- **FFT.h / FFT.cc** - Original FFT implementation (now wrapped by interface)

### Design Principles

1. **Drop-in replacement**: All interface functions maintain the same semantics as the original FFT functions
2. **Backend abstraction**: The interface supports multiple backends (CPU, GPU) with runtime selection
3. **Zero overhead**: When using CPU backend, the interface is a thin wrapper with negligible overhead
4. **Future-proof**: Designed to easily integrate cuFFT for GPU acceleration

## API Functions

### Core FFT Operations

```cpp
// Integer array ↔ FFT domain
void FFT_Interface_IntToFFT(CC_t * f_FFT, const long int * const f);
void FFT_Interface_FFTToInt(long int * const f, CC_t const * const f_fft);

// Real array ← FFT domain  
void FFT_Interface_FFTToReal(double * const f, CC_t const * const f_fft);

// NTL polynomial ↔ FFT domain
void FFT_Interface_ZZXToFFT(CC_t * f_FFT, const ZZX f);
void FFT_Interface_FFTToZZX(ZZX& f, CC_t const * const f_FFT);
```

### Backend Management

```cpp
// Initialize FFT backend (optional - auto-initializes with CPU if not called)
int FFT_Interface_Init(FFT_Backend_Type backend);

// Cleanup resources
void FFT_Interface_Cleanup();

// Query backend
FFT_Backend_Type FFT_Interface_GetBackend();
int FFT_Interface_IsBackendAvailable(FFT_Backend_Type backend);
```

### Performance Hints (for future GPU optimization)

```cpp
// Signal that multiple FFT operations will be performed
// Enables batching optimizations on GPU
void FFT_Interface_BeginBatch();
void FFT_Interface_EndBatch();
```

## Usage

### Current Usage (CPU)

The interface automatically initializes with the CPU backend on first use. No explicit initialization is required:

```cpp
CC_t fft_data[N0];
long int input[N0];

// Automatically uses CPU backend
FFT_Interface_IntToFFT(fft_data, input);
```

### Future GPU Usage

When GPU support is added, you can explicitly initialize:

```cpp
// Initialize GPU backend
if (FFT_Interface_Init(FFT_BACKEND_GPU) == 0) {
    // GPU backend successfully initialized
} else {
    // Falls back to CPU
}

// Use batching for multiple operations
FFT_Interface_BeginBatch();
for (int i = 0; i < many_ops; i++) {
    FFT_Interface_IntToFFT(...);
}
FFT_Interface_EndBatch();

// Cleanup when done
FFT_Interface_Cleanup();
```

## Integration Status

All FFT calls in the codebase have been migrated to use the interface:

### Scheme.cc
- ✅ `CompleteMSK()` - Converting master secret key to FFT
- ✅ `CompleteMPK()` - Converting master public key to FFT
- ✅ `IBE_Encrypt()` - FFT operations for encryption
- ✅ `IBE_Decrypt()` - FFT operations for decryption
- ✅ `Extract_Bench()` - Test/benchmark FFT conversions
- ✅ `Encrypt_Bench()` - Test/benchmark FFT conversions

### Algebra.cc
- ✅ `FastReductionCoefficient()` - FFT-based polynomial operations
- ✅ `GS_Norm()` - Gram-Schmidt normalization with FFT

## Adding GPU Support

To add cuFFT support in the future:

1. **Add CUDA compilation** to Makefile:
   ```makefile
   NVCC=nvcc
   CUDA_FLAGS=-O3 -arch=sm_70
   FFT_Interface.o: FFT_Interface.cu
       $(NVCC) $(CUDA_FLAGS) -c FFT_Interface.cu
   ```

2. **Implement GPU backend** in FFT_Interface.cc/cu:
   - Create cuFFT plans in `FFT_Interface_Init(FFT_BACKEND_GPU)`
   - Implement device memory management
   - Convert CC_t arrays to cuFFT-compatible format
   - Handle data transfers between host and device
   - Utilize batching in BeginBatch/EndBatch

3. **Add availability check**:
   ```cpp
   int FFT_Interface_IsBackendAvailable(FFT_Backend_Type backend) {
       if (backend == FFT_BACKEND_GPU) {
           #ifdef __CUDACC__
           int device_count;
           cudaGetDeviceCount(&device_count);
           return device_count > 0;
           #else
           return 0;
           #endif
       }
       return 1; // CPU always available
   }
   ```

## Testing

All existing tests pass with the interface:
- ✅ Key generation works correctly
- ✅ Extraction tests pass (100 identities)
- ✅ Encryption/decryption tests pass (100 messages)
- ✅ Performance timing statistics work

## Performance Notes

- Current CPU backend has minimal overhead (thin wrapper)
- GPU backend will benefit from:
  - Batching multiple FFT operations
  - Parallel execution for independent transforms
  - Reduced data transfer with persistent GPU memory
  - cuFFT's highly optimized implementation

## Next Steps

1. Profile to identify FFT hotspots for GPU acceleration
2. Implement cuFFT backend
3. Add benchmarking to compare CPU vs GPU performance
4. Optimize data transfers (pinned memory, streams)
5. Consider implementing async operations for better pipelining
