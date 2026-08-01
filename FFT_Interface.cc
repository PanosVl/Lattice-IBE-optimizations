#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "FFT_Interface.h"
#include "FFT_CUDA_Backend.h"
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
        if (FFT_CUDA_Init() == 0) {
            g_current_backend = FFT_BACKEND_GPU;
            g_initialized = true;
            return 0;
        }
        cerr << "Warning: GPU backend initialization failed, falling back to CPU" << endl;
        g_current_backend = FFT_BACKEND_CPU;
        g_initialized = true;
        return 1;
    }
    
    g_current_backend = backend;
    g_initialized = true;
    return 0;
}

void FFT_Interface_Cleanup() {
    if (g_current_backend == FFT_BACKEND_GPU) {
        FFT_CUDA_Cleanup();
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
        return FFT_CUDA_IsAvailable();
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
        if (FFT_CUDA_IntToFFT(f_FFT, f) != 0) {
            cerr << "Warning: GPU IntToFFT failed, using CPU implementation" << endl;
            MyIntFFT(f_FFT, f);
        }
    }
}

void FFT_Interface_IntToFFT_Batch(CC_t * f_FFT, const long int * const f, unsigned int batch_count) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }

    if (batch_count == 0) {
        return;
    }

    if (g_current_backend == FFT_BACKEND_CPU) {
        for (unsigned int i = 0; i < batch_count; ++i) {
            MyIntFFT(f_FFT + i * N0, f + i * N0);
        }
    } else {
        if (FFT_CUDA_IntToFFT_Batch(f_FFT, f, batch_count) != 0) {
            cerr << "Warning: GPU IntToFFT batch failed, using CPU implementation" << endl;
            for (unsigned int i = 0; i < batch_count; ++i) {
                MyIntFFT(f_FFT + i * N0, f + i * N0);
            }
        }
    }
}

void FFT_Interface_FFTToInt(long int * const f, CC_t const * const f_fft) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }
    
    if (g_current_backend == FFT_BACKEND_CPU) {
        MyIntReverseFFT(f, f_fft);
    } else {
        if (FFT_CUDA_FFTToInt(f, f_fft) != 0) {
            cerr << "Warning: GPU FFTToInt failed, using CPU implementation" << endl;
            MyIntReverseFFT(f, f_fft);
        }
    }
}

void FFT_Interface_FFTToInt_Batch(long int * const f, CC_t const * const f_fft, unsigned int batch_count) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }

    if (batch_count == 0) {
        return;
    }

    if (g_current_backend == FFT_BACKEND_CPU) {
        for (unsigned int i = 0; i < batch_count; ++i) {
            MyIntReverseFFT(f + i * N0, f_fft + i * N0);
        }
    } else {
        if (FFT_CUDA_FFTToInt_Batch(f, f_fft, batch_count) != 0) {
            cerr << "Warning: GPU FFTToInt batch failed, using CPU implementation" << endl;
            for (unsigned int i = 0; i < batch_count; ++i) {
                MyIntReverseFFT(f + i * N0, f_fft + i * N0);
            }
        }
    }
}

void FFT_Interface_FFTToReal(double * const f, CC_t const * const f_fft) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }
    
    if (g_current_backend == FFT_BACKEND_CPU) {
        MyRealReverseFFT(f, f_fft);
    } else {
        if (FFT_CUDA_FFTToReal(f, f_fft) != 0) {
            cerr << "Warning: GPU FFTToReal failed, using CPU implementation" << endl;
            MyRealReverseFFT(f, f_fft);
        }
    }
}

void FFT_Interface_ZZXToFFT(CC_t * f_FFT, const ZZX f) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }
    
    if (g_current_backend == FFT_BACKEND_CPU) {
        ZZXToFFT(f_FFT, f);
    } else {
        // Current CUDA path supports integer/real transforms only; keep semantic parity.
        ZZXToFFT(f_FFT, f);
    }
}

void FFT_Interface_FFTToZZX(ZZX& f, CC_t const * const f_FFT) {
    if (!g_initialized) {
        FFT_Interface_Init(FFT_BACKEND_CPU);
    }
    
    if (g_current_backend == FFT_BACKEND_CPU) {
        FFTToZZX(f, f_FFT);
    } else {
        // Current CUDA path supports integer/real transforms only; keep semantic parity.
        FFTToZZX(f, f_FFT);
    }
}
