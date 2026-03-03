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
