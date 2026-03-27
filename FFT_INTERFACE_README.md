# FFT Interface Abstraction Layer

## Overview

This document describes the FFT abstraction layer created to enable future GPU acceleration using cuFFT. The interface provides a uniform API that abstracts the underlying FFT implementation, allowing seamless switching between CPU and GPU backends.

## Architecture

### Files

- **FFT_Interface.h** - Interface declarations and API
- **FFT_Interface.cc** - Current CPU-based implementation
- **FFT.h / FFT.cc** - Original FFT implementation (now wrapped by interface)

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

## Usage

### Current Usage (CPU)

The interface automatically initializes with the CPU backend on first use. No explicit initialization is required:

```cpp
CC_t fft_data[N0];
long int input[N0];

// Automatically uses CPU backend
FFT_Interface_IntToFFT(fft_data, input);
```

## CUDA Backend (First Milestone)

The project now includes an initial CUDA/cuFFT backend scaffold in:

- `FFT_CUDA_Backend.h`
- `FFT_CUDA_Backend.cu`

This first step provides CUDA execution for:

- `FFT_Interface_IntToFFT`
- `FFT_Interface_FFTToInt`
- `FFT_Interface_FFTToReal`

The `ZZX` conversion functions currently keep CPU behavior while the CUDA path is expanded.

### Build with CUDA

Default build remains CPU-only:

```bash
make
```

To enable CUDA backend compilation (requires `nvcc`, CUDA runtime, and cuFFT):

```bash
make USE_CUDA=1
```

### Runtime Behavior

- If GPU backend init succeeds, interface calls use CUDA where implemented.
- If initialization fails, the interface falls back to CPU.
- If a CUDA operation fails at runtime, that call falls back to CPU and continues safely.

## Implementation Status (Uncommitted Milestone)

### Files Added/Updated

- `FFT_CUDA_Backend.h`: CUDA backend API surface used by the interface.
- `FFT_CUDA_Backend.cc`: wrapper/stub dispatch layer (`USE_CUDA` vs non-CUDA builds).
- `FFT_CUDA_Backend.cu`: cuFFT implementation and CUDA kernels.
- `FFT_Interface.cc`: GPU init/availability integration and per-call CPU fallback.
- `Makefile`: optional CUDA build path via `USE_CUDA=1`.

### Operation Coverage

- CUDA implemented:
	- `FFT_Interface_IntToFFT`
	- `FFT_Interface_FFTToInt`
	- `FFT_Interface_FFTToReal`
- CPU-only for now:
	- `FFT_Interface_ZZXToFFT`
	- `FFT_Interface_FFTToZZX`

## Build/Run Prerequisites

For CUDA builds (`make USE_CUDA=1`), the environment must provide:

- `nvcc` in `PATH`
- CUDA runtime libraries (for `-lcudart`)
- cuFFT library (for `-lcufft`)
- A visible CUDA-capable GPU at runtime

Without `nvcc`, CPU mode still builds normally with:

```bash
make
```

## Queue-Based Validation Checklist (e.g., kelvin2)

Use this sequence when moving from a non-CUDA dev machine to a GPU queue system.

```bash
# 1) Load required modules according to cluster policy
module avail cuda
module load cuda

# 2) Verify toolchain and GPU visibility
which nvcc
nvcc --version
nvidia-smi

# 3) Build with CUDA path enabled
make clean
make USE_CUDA=1
```

Recommended runtime smoke checks in a GPU job:

- `FFT_Interface_IsBackendAvailable(FFT_BACKEND_GPU)` returns true.
- `FFT_Interface_Init(FFT_BACKEND_GPU)` returns `0` (no fallback).
- Roundtrip consistency for representative vectors:
	- `IntToFFT -> FFTToInt`
	- `IntToFFT -> FFTToReal`
- Compare CUDA results against CPU baseline on the same inputs.

If init fails or operations return non-zero, interface-level CPU fallback should preserve correctness while indicating the GPU issue through warning logs.
