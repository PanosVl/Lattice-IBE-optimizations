# Timing Instrumentation for Lattice-IBE

## Overview
This document describes the timing instrumentation added to the Lattice-IBE implementation to establish CPU performance baselines for the main cryptographic operations.

## Instrumented Functions

### High-Level Operations
The following main IBE operations are instrumented with complete timing:

1. **Keygen** - Master key generation
2. **IBE_Extract** - User private key extraction from identity
3. **IBE_Encrypt** - Message encryption
4. **IBE_Decrypt** - Ciphertext decryption

### Low-Level Operations
Within the expensive functions, detailed timing was added for:

1. **FFT Operations**:
   - `MyIntFFT()` - Integer to FFT domain conversion
   - `MyIntReverseFFT()` - FFT to integer domain conversion
   - Tracked in: IBE_Encrypt (4 calls), IBE_Decrypt (2 calls)

2. **Sampling Operations**:
   - `Sample4()` - Discrete Gaussian sampling
   - Tracked in: GPV function (called by IBE_Extract)
   - Approximately 2*N0 = 1024 samples per extraction

## Implementation Details

### Timing Variables
Global static variables track cumulative time and call counts:
- `g_keygen_time`, `g_keygen_count`
- `g_extract_time`, `g_extract_count`
- `g_encrypt_time`, `g_encrypt_count`
- `g_decrypt_time`, `g_decrypt_count`
- `g_fft_time`, `g_fft_count`
- `g_sampling_time`, `g_sampling_count`

### API Functions
Two new functions are available in Scheme.h:

```cpp
void print_timing_stats();  // Display timing breakdown with percentages
void reset_timing_stats();  // Reset all counters (if needed for multiple runs)
```

## Usage

The timing statistics are automatically collected during normal program execution and printed at the end via the call to `print_timing_stats()` in IBE.cc.

To run and see timing statistics:
```bash
make clean && make
./IBE
```

## Notes

- Timing uses `std::chrono::high_resolution_clock` for accurate measurements
- The instrumentation has minimal overhead as timers are only called at function boundaries
- Sampling timing is measured at the innermost loop level in GPV() where Sample4() is called
- FFT timing measures both forward and reverse FFT operations separately
