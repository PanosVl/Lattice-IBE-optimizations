#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "FFT_Interface.h"
#include "FFT.h"
#include "params.h"

using namespace std;

// ============================================================================
// Backend state
// ============================================================================

static FFT_Backend_Type g_current_backend = FFT_BACKEND_CPU;
static bool g_initialized = false;
static bool g_in_batch = false;

// ============================================================================
// Backend management
// ============================================================================

int FFT_Interface_Init(FFT_Backend_Type backend) {
    if (backend == FFT_BACKEND_GPU) {
        // GPU backend not yet implemented
        cerr << "Warning: GPU backend not yet available, falling back to CPU" << endl;
        g_current_backend = FFT_BACKEND_CPU;
        g_initialized = true;
        return 1; // Indicate fallback
    }
    
    g_current_backend = backend;
    g_initialized = true;
    return 0;
}

void FFT_Interface_Cleanup() {
    if (g_current_backend == FFT_BACKEND_GPU) {
        // Cleanup GPU resources (to be implemented)
    }
    g_initialized = false;
}

FFT_Backend_Type FFT_Interface_GetBackend() {
    return g_current_backend;
}

int FFT_Interface_IsBackendAvailable(FFT_Backend_Type backend) {
    if (backend == FFT_BACKEND_CPU) {
        return 1; // CPU always available
    } else if (backend == FFT_BACKEND_GPU) {
        // Check for CUDA/GPU availability (to be implemented)
        return 0;
    }
    return 0;
}

// ============================================================================
// Batch processing hints
// ============================================================================

void FFT_Interface_BeginBatch() {
    g_in_batch = true;
    // For CPU backend, this is just a hint (no-op)
    // For GPU backend, this could trigger memory pinning or stream setup
}

void FFT_Interface_EndBatch() {
    g_in_batch = false;
    // For GPU backend, this would synchronize streams
}

// ============================================================================
// FFT operations - CPU backend implementation
// ============================================================================

void FFT_Interface_IntToFFT(CC_t * f_FFT, const long int * const f) {
    // Auto-initialize with CPU backend if not initialized
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }
    
    if (g_current_backend == FFT_BACKEND_CPU) {
        MyIntFFT(f_FFT, f);
    } else {
        // GPU implementation would go here
        cerr << "Error: GPU backend called but not implemented" << endl;
        abort();
    }
}

void FFT_Interface_FFTToInt(long int * const f, CC_t const * const f_fft) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }
    
    if (g_current_backend == FFT_BACKEND_CPU) {
        MyIntReverseFFT(f, f_fft);
    } else {
        // GPU implementation would go here
        cerr << "Error: GPU backend called but not implemented" << endl;
        abort();
    }
}

void FFT_Interface_FFTToReal(double * const f, CC_t const * const f_fft) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }
    
    if (g_current_backend == FFT_BACKEND_CPU) {
        MyRealReverseFFT(f, f_fft);
    } else {
        // GPU implementation would go here
        cerr << "Error: GPU backend called but not implemented" << endl;
        abort();
    }
}

void FFT_Interface_ZZXToFFT(CC_t * f_FFT, const ZZX f) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }
    
    if (g_current_backend == FFT_BACKEND_CPU) {
        ZZXToFFT(f_FFT, f);
    } else {
        // GPU implementation would go here
        cerr << "Error: GPU backend called but not implemented" << endl;
        abort();
    }
}

void FFT_Interface_FFTToZZX(ZZX& f, CC_t const * const f_FFT) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }
    
    if (g_current_backend == FFT_BACKEND_CPU) {
        FFTToZZX(f, f_FFT);
    } else {
        // GPU implementation would go here
        cerr << "Error: GPU backend called but not implemented" << endl;
        abort();
    }
}
