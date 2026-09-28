# Timing Instrumentation for Lattice-IBE

## Overview

The instrumentation records elapsed host time for the main IBE protocol operations and for selected computational components. It is intended for comparative profiling and bottleneck analysis, not as a replacement for a fully controlled end-to-end latency benchmark.

## Instrumented Operations

### Protocol-level operations

The following operations are timed as complete phases:

1. **Keygen** - Master-key generation
2. **IBE_Extract** - Private-key extraction for an identity
3. **IBE_Encrypt** - Encryption of one message
4. **IBE_Decrypt** - Decryption of one ciphertext

Each phase accumulates total elapsed time and invocation count. The report also displays the average time per call and the phase's share of the combined instrumented protocol time.

### Component-level operations

The detailed report includes timers for:

- basis generation and quotient computation during key generation;
- Gaussian sampling and GPV overhead during extraction;
- forward and reverse FFT processing during encryption and decryption; and
- modular arithmetic during encryption and decryption.

The FFT counter records the number of transforms represented by each timed block. In the current encryption path, each batched forward or reverse block represents two transforms; decryption records one transform per block.

## Timing semantics

The detailed timers use `std::chrono::high_resolution_clock` and record elapsed host-side duration. The clock is not a process-CPU timer and should not be described as measuring CPU time exclusively. On the target platform, it should be treated as an elapsed-time measurement made by the host process.

When the CUDA backend is selected, the host timer surrounds the call into the CUDA FFT backend. Consequently, the reported FFT duration includes the host-visible duration of that backend call, including any synchronous transfers, kernel launches, cuFFT execution, and result handling performed before the call returns. The current checked-out backend does not provide a separate CUDA-event measurement, so the report does not isolate GPU device time from host elapsed time.

There is no dedicated wall-clock timer around the complete benchmark in the timing report. The protocol and component values are host elapsed durations captured by the same `high_resolution_clock`; they should therefore be compared only with measurements collected using the same timer and experimental conditions. A separate synchronized wall-clock benchmark is required for a user-visible end-to-end latency claim.

## Nested measurements and percentages

Component timers are nested within protocol-level timers. Where the report provides an `Other` category, it estimates the residual parent time after subtracting the explicitly measured components. The displayed percentages are therefore local to their parent phase and should not be added across the complete report. GPU work, when enabled, is represented through the host-side FFT interval and is not a separate percentage category in the current implementation.

## Recorded counters

The implementation maintains cumulative durations and call counts for:

- `g_keygen_time`, `g_keygen_count`
- `g_extract_time`, `g_extract_count`
- `g_encrypt_time`, `g_encrypt_count`
- `g_decrypt_time`, `g_decrypt_count`
- `g_fft_time`, `g_fft_count`
- `g_sampling_time`, `g_sampling_count`
- `g_basis_gen_time`, `g_basis_gen_count`
- `g_quotient_time`, `g_quotient_count`
- `g_gpv_overhead_time`, `g_gpv_overhead_count`
- `g_modarith_time`, `g_modarith_count`

The public timing functions are:

```cpp
void print_timing_stats();  // Print the timing breakdown
void reset_timing_stats();  // Reset accumulated durations and counters
```

## Usage

Timing statistics are collected during normal execution and printed by `IBE.cc` through `print_timing_stats()`.

CPU-backend run:

```bash
make clean && make
./IBE
```

CUDA-backend run on a CUDA-enabled system:

```bash
make clean && make USE_CUDA=1 USE_KELVIN2=1 -j$(nproc)
./IBE
```

## Reporting and speedup

Report absolute phase time, average time per call, and phase-local component shares. Calculate speedup only between measurements with the same timer semantics, parameter set, iteration count, backend, and hardware. Because the current report uses host elapsed time for both CPU and CUDA-backed calls, it can support comparison of end-to-end host-visible phase duration, but it cannot support a claim about isolated GPU execution time. Such a claim requires CUDA-event instrumentation or another explicitly synchronized device-side measurement.
