#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <chrono>
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <NTL/mat_ZZ.h>
#include <gmp.h>

#include "Sampling.h"
#include "params.h"
#include "FFT.h"
#include "FFT_Interface.h"
#include "Random.h"
#include "Algebra.h"

using namespace std;
using namespace NTL;

const ZZX phi = Cyclo();

// Timing variables for performance profiling
static double g_keygen_time = 0.0;
static double g_extract_time = 0.0;
static double g_encrypt_time = 0.0;
static double g_decrypt_time = 0.0;
static double g_fft_time = 0.0;
static double g_sampling_time = 0.0;
static double g_basis_gen_time = 0.0;
static double g_quotient_time = 0.0;
static double g_gpv_overhead_time = 0.0;
static double g_modarith_time = 0.0;
static long long g_keygen_count = 0;
static long long g_extract_count = 0;
static long long g_encrypt_count = 0;
static long long g_decrypt_count = 0;
static long long g_fft_count = 0;
static long long g_sampling_count = 0;
static long long g_basis_gen_count = 0;
static long long g_quotient_count = 0;
static long long g_gpv_overhead_count = 0;
static long long g_modarith_count = 0;

void print_timing_stats() {
    cout << "\n=== IBE Performance Statistics ===" << endl;
    if (g_keygen_count > 0) {
       cout << "Keygen:    " << g_keygen_time << "s (" << g_keygen_count << " calls, avg: " 
           << (g_keygen_time / g_keygen_count * 1000) << "ms, " 
           << ((g_keygen_time / (g_keygen_time + g_extract_time + g_encrypt_time + g_decrypt_time)) * 100.0)
           << "% of total)" << endl;
    }
    if (g_extract_count > 0) {
       cout << "Extract:   " << g_extract_time << "s (" << g_extract_count << " calls, avg: " 
           << (g_extract_time / g_extract_count * 1000) << "ms, " 
           << ((g_extract_time / (g_keygen_time + g_extract_time + g_encrypt_time + g_decrypt_time)) * 100.0)
           << "% of total)" << endl;
    }
    if (g_encrypt_count > 0) {
       cout << "Encrypt:   " << g_encrypt_time << "s (" << g_encrypt_count << " calls, avg: " 
           << (g_encrypt_time / g_encrypt_count * 1000) << "ms, " 
           << ((g_encrypt_time / (g_keygen_time + g_extract_time + g_encrypt_time + g_decrypt_time)) * 100.0)
           << "% of total)" << endl;
    }
    if (g_decrypt_count > 0) {
       cout << "Decrypt:   " << g_decrypt_time << "s (" << g_decrypt_count << " calls, avg: " 
           << (g_decrypt_time / g_decrypt_count * 1000) << "ms, " 
           << ((g_decrypt_time / (g_keygen_time + g_extract_time + g_encrypt_time + g_decrypt_time)) * 100.0)
           << "% of total)" << endl;
    }
    
    double total_time = g_keygen_time + g_extract_time + g_encrypt_time + g_decrypt_time;
    if (total_time > 0) {
        cout << "\nDetailed Breakdown:" << endl;
       if (g_keygen_time > 0) {
          double keygen_remaining = g_keygen_time - g_basis_gen_time - g_quotient_time;
          if (keygen_remaining < 0.0) {
             keygen_remaining = 0.0;
          }
          cout << "  Keygen:" << endl;
          cout << "    Basis Gen:   " << g_basis_gen_time << "s (" << (g_basis_gen_time / g_keygen_time) * 100.0
              << "% of keygen, " << (g_basis_gen_time / total_time) * 100.0 << "% of total)" << endl;
          cout << "    Quotient:    " << g_quotient_time << "s (" << (g_quotient_time / g_keygen_time) * 100.0
              << "% of keygen, " << (g_quotient_time / total_time) * 100.0 << "% of total)" << endl;
          cout << "    Other:       " << keygen_remaining << "s (" << (keygen_remaining / g_keygen_time) * 100.0
              << "% of keygen, " << (keygen_remaining / total_time) * 100.0 << "% of total)" << endl;
        }
       if (g_extract_time > 0) {
          double extract_remaining = g_extract_time - g_sampling_time - g_gpv_overhead_time;
          if (extract_remaining < 0.0) {
             extract_remaining = 0.0;
          }
          cout << "  Extract:" << endl;
          cout << "    Sampling:    " << g_sampling_time << "s (" << g_sampling_count << " calls, " 
              << (g_sampling_time / g_extract_time) * 100.0 << "% of extract, "
              << (g_sampling_time / total_time) * 100.0 << "% of total)" << endl;
          cout << "    GPV Overhead: " << g_gpv_overhead_time << "s (" << g_gpv_overhead_count << " calls, " 
              << (g_gpv_overhead_time / g_extract_time) * 100.0 << "% of extract, "
              << (g_gpv_overhead_time / total_time) * 100.0 << "% of total)" << endl;
          cout << "    Other:       " << extract_remaining << "s (" << (extract_remaining / g_extract_time) * 100.0
              << "% of extract, " << (extract_remaining / total_time) * 100.0 << "% of total)" << endl;
        }
       if ((g_encrypt_time + g_decrypt_time) > 0) {
          double crypto_time = g_encrypt_time + g_decrypt_time;
          double crypto_remaining = crypto_time - g_fft_time - g_modarith_time;
          if (crypto_remaining < 0.0) {
             crypto_remaining = 0.0;
          }
          cout << "  Encrypt+Decrypt:" << endl;
          cout << "    FFT:         " << g_fft_time << "s (" << g_fft_count << " calls, " 
              << (g_fft_time / crypto_time) * 100.0 << "% of encrypt+decrypt, "
              << (g_fft_time / total_time) * 100.0 << "% of total)" << endl;
          cout << "    Mod Arith:   " << g_modarith_time << "s (" << g_modarith_count << " calls, " 
              << (g_modarith_time / crypto_time) * 100.0 << "% of encrypt+decrypt, "
              << (g_modarith_time / total_time) * 100.0 << "% of total)" << endl;
          cout << "    Other:       " << crypto_remaining << "s (" << (crypto_remaining / crypto_time) * 100.0
              << "% of encrypt+decrypt, " << (crypto_remaining / total_time) * 100.0 << "% of total)" << endl;
        }
        cout << "\nTotal:     " << total_time << "s" << endl;
    }
    cout << "=================================\n" << endl;
}

void reset_timing_stats() {
    g_keygen_time = 0.0;
    g_extract_time = 0.0;
    g_encrypt_time = 0.0;
    g_decrypt_time = 0.0;
    g_fft_time = 0.0;
    g_sampling_time = 0.0;
    g_basis_gen_time = 0.0;
    g_quotient_time = 0.0;
    g_gpv_overhead_time = 0.0;
    g_modarith_time = 0.0;
    g_keygen_count = 0;
    g_extract_count = 0;
    g_encrypt_count = 0;
    g_decrypt_count = 0;
    g_fft_count = 0;
    g_sampling_count = 0;
    g_basis_gen_count = 0;
    g_quotient_count = 0;
    g_gpv_overhead_count = 0;
    g_modarith_count = 0;
}


//==============================================================================
//Generates from parameters N and q :
// - a public key : polynomial h
// - a private key : polynomials f,g,F,G
//==============================================================================
void Keygen(ZZ_pX& PublicKey, ZZX* PrivateKey)
{
    auto start_time = chrono::high_resolution_clock::now();
    
    ZZ SqNorm;
    ZZX f,g,F,G;

    SqNorm = conv<ZZ>(1.36*q0/2);

    auto basis_start = chrono::high_resolution_clock::now();
    GenerateBasis(f, g, F, G, SqNorm);
    auto basis_end = chrono::high_resolution_clock::now();
    chrono::duration<double> basis_elapsed = basis_end - basis_start;
    g_basis_gen_time += basis_elapsed.count();
    g_basis_gen_count++;
    
    PrivateKey[0] = f;
    PrivateKey[1] = g;
    PrivateKey[2] = F;
    PrivateKey[3] = G;

    for(unsigned int i=0; i<4; i++)
    {
            PrivateKey[i].SetLength(N0);
    }

    auto quotient_start = chrono::high_resolution_clock::now();
    PublicKey = Quotient(f, g);
    auto quotient_end = chrono::high_resolution_clock::now();
    chrono::duration<double> quotient_elapsed = quotient_end - quotient_start;
    g_quotient_time += quotient_elapsed.count();
    g_quotient_count++;
    
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    g_keygen_time += elapsed.count();
    g_keygen_count++;
}

//==============================================================================
//Computes the private basis B from private key PrivateKey and parameter N
//==============================================================================
void CompletePrivateKey(mat_ZZ& B, const ZZX * const PrivateKey)
{
    ZZX f,g,F,G;
    f = PrivateKey[0];
    g = PrivateKey[1];
    F = PrivateKey[2];
    G = PrivateKey[3];

    f = -f;
    F = -F;

    B = BasisFromPolynomials(g, f, G, F);
}





void GPV(RR_t * v, const RR_t * const c, const RR_t s, const MSK_Data * const MSKD)
{
    auto gpv_start = chrono::high_resolution_clock::now();

    int i;
    unsigned j;
    RR_t ci[2*N0], zi, cip, sip, aux;

    for(j=0; j<2*N0;j++)
    {
        ci[j] = c[j];
    }

    for(j=0; j<2*N0; j++)
    {

    }    

    for(i=2*N0-1; i>=0; i--)
    {
        auto iter_start = chrono::high_resolution_clock::now();
        aux = (MSKD->GS_Norms)[i];
        cip = DotProduct(ci, MSKD->Bstar[i])/(aux*aux);
        sip = s/aux;
        auto iter_end = chrono::high_resolution_clock::now();
        
        auto sampling_start = chrono::high_resolution_clock::now();
        zi = Sample4(cip, sip*PiPrime);
        auto sampling_end = chrono::high_resolution_clock::now();
        chrono::duration<double> sampling_elapsed = sampling_end - sampling_start;
        g_sampling_time += sampling_elapsed.count();
        g_sampling_count++;

        auto update_start = chrono::high_resolution_clock::now();
        for(j=0; j<2*N0; j++)
        {
            ci[j] -= zi*(MSKD->B)[i][j];
        }
        auto update_end = chrono::high_resolution_clock::now();
        
        chrono::duration<double> iter_elapsed = iter_end - iter_start;
        chrono::duration<double> update_elapsed = update_end - update_start;
        g_gpv_overhead_time += iter_elapsed.count() + update_elapsed.count();
    }
    g_gpv_overhead_count++;
    
    auto gpv_end = chrono::high_resolution_clock::now();
    chrono::duration<double> gpv_elapsed = gpv_end - gpv_start;

    for(j=0; j<2*N0; j++)
    {
        v[j] = c[j] - ci[j];
    }

}



//==============================================================================
//==============================================================================
//                            MAIN PROGRAMS
//==============================================================================
//==============================================================================


void CompleteMSK(MSK_Data * MSKD, ZZX * MSK)
{
    unsigned int i, j;
    mat_ZZ B0;

    for(i=0; i<4; i++)
    {
        MSKD->PrK[i] = MSK[i];
        ZZXToFFT(MSKD->PrK_fft[i], MSK[i]);
    }

    CompletePrivateKey(B0, MSK);

    for(i=0; i<2*N0; i++)
    {
        for(j=0; j<2*N0; j++)
        {
            MSKD->B[i][j] = ( (RR_t) conv<double>(B0[i][j]) );
        }
    }

    for(i=0; i<1; i++)
    {
        FastMGS(MSKD->Bstar, MSKD->B);
    }

    for(i=0; i<2*N0; i++)
    {
        MSKD->GS_Norms[i] = sqrt( DotProduct(MSKD->Bstar[i], MSKD->Bstar[i]) );
    }

    MSKD->sigma = 2*MSKD->GS_Norms[0];

}



void CompleteMPK(MPK_Data * MPKD, ZZ_pX MPK)
{
    MPKD->h = MPK;
    FFT_Interface_ZZXToFFT(MPKD->h_FFT, conv<ZZX>(MPK));
}



void IBE_Extract(ZZX SK_id[2], vec_ZZ id, const MSK_Data * const MSKD)
{
    auto start_time = chrono::high_resolution_clock::now();
    
    unsigned int i;
    RR_t c[2*N0], sk[2*N0], sigma;
    ZZX f,g,aux;

    f = MSKD -> PrK[0];
    g = MSKD -> PrK[1];
    sigma = MSKD->sigma;
    SK_id[0].SetLength(N0);
    SK_id[1].SetLength(N0);

    for(i=0;i<N0;i++)
    {
        c[i] = ((RR_t) conv<double>(id[i])) ;
        c[i+N0] = 0;
    }

    GPV(sk, c, sigma, MSKD);

    for(i=0; i<N0; i++)
    {
        sk[i] = c[i] - sk[i];
        sk[i+N0] = - sk[i+N0];
    }

    for(i=0; i<N0; i++)
    {
        SK_id[0][i] = sk[i];
        SK_id[1][i] = sk[i+N0];
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    g_extract_time += elapsed.count();
    g_extract_count++;
}


unsigned long IBE_Verify_Key(const ZZX SK_id[2], const vec_ZZ id, const MSK_Data * const MSKD)
{
    unsigned int i;
    ZZX f,g,t,aux;

    f = MSKD -> PrK[0];
    g = MSKD -> PrK[1];
    
    t = conv<ZZX>(id);
    aux = ((SK_id[0] - t)*f + g*SK_id[1])%phi;

    for(i=0; i<N0; i++)
    {
        aux[i] %= q1;
    }

    if( IsZero(aux) != 0)
    {
        cout << "The signature (s1,s2) doesn't verify the required equality [ (s1 - t)*f + g*s2 = 0 ] !\nActually, (s1 - t)*f + g*s2 = " << aux << endl << endl;
    }
    return IsZero(aux);
}


void IBE_Encrypt(long C[2][N0], const long m[N0], const long id0[N0], const MPK_Data * const MPKD)
{
    auto start_time = chrono::high_resolution_clock::now();

    unsigned long i;
    long r[N0], e1[N0], e2[N0];
    CC_t r_FFT[N0], t_FFT[N0], aux1_FFT[N0], aux2_FFT[N0];

    for(i=0; i<N0; i++)
    {
        e1[i] = (rand()%3) - 1;
        e2[i] = (rand()%3) - 1;
        r[i] = (rand()%3) - 1;
    }

    auto fft_start = chrono::high_resolution_clock::now();
    FFT_Interface_IntToFFT(r_FFT, r);
    FFT_Interface_IntToFFT(t_FFT, id0);
    auto fft_end = chrono::high_resolution_clock::now();
    chrono::duration<double> fft_elapsed = fft_end - fft_start;
    g_fft_time += fft_elapsed.count();
    g_fft_count += 2;

    for(i=0; i<N0; i++)
    {
        aux1_FFT[i] = r_FFT[i]*((MPKD->h_FFT)[i]);
        aux2_FFT[i] = r_FFT[i]*t_FFT[i];
    }

    fft_start = chrono::high_resolution_clock::now();
    FFT_Interface_FFTToInt(C[0], aux1_FFT);
    FFT_Interface_FFTToInt(C[1], aux2_FFT);
    fft_end = chrono::high_resolution_clock::now();
    fft_elapsed = fft_end - fft_start;
    g_fft_time += fft_elapsed.count();
    g_fft_count += 2;

    auto modarith_start = chrono::high_resolution_clock::now();
    for(i=0; i<N0; i++)
    {
        C[0][i] = (C[0][i] + e1[i]               + q0/2)%q0 - (q0/2);
        C[1][i] = (C[1][i] + e2[i] + (q0/2)*m[i] + q0/2)%q0 - (q0/2);
    }
    auto modarith_end = chrono::high_resolution_clock::now();
    chrono::duration<double> modarith_elapsed = modarith_end - modarith_start;
    g_modarith_time += modarith_elapsed.count();
    g_modarith_count++;

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    g_encrypt_time += elapsed.count();
    g_encrypt_count++;
}


void IBE_Decrypt(long message[N0], const long C[2][N0], const CC_t * const SKid_FFT)
{
    auto start_time = chrono::high_resolution_clock::now();
    
    unsigned int i;
    CC_t c0_FFT[N0], aux_FFT[N0];

    auto fft_start = chrono::high_resolution_clock::now();
    FFT_Interface_IntToFFT(c0_FFT, C[0]);
    auto fft_end = chrono::high_resolution_clock::now();
    chrono::duration<double> fft_elapsed = fft_end - fft_start;
    g_fft_time += fft_elapsed.count();
    g_fft_count++;

    for(i=0; i<N0; i++)
    {
        aux_FFT[i] = c0_FFT[i]*SKid_FFT[i];
    }

    fft_start = chrono::high_resolution_clock::now();
    FFT_Interface_FFTToInt(message, aux_FFT);
    fft_end = chrono::high_resolution_clock::now();
    fft_elapsed = fft_end - fft_start;
    g_fft_time += fft_elapsed.count();
    g_fft_count++;

    auto modarith_start = chrono::high_resolution_clock::now();
    for(i=0; i<N0; i++)
    {
        message[i] = C[1][i] - message[i];
        message[i] = ((unsigned long)(message[i] ))%q0;
        message[i] = (message[i] + (q0>>2) )/(q0>>1);
        message[i] %= 2;
    }
    auto modarith_end = chrono::high_resolution_clock::now();
    chrono::duration<double> modarith_elapsed = modarith_end - modarith_start;
    g_modarith_time += modarith_elapsed.count();
    g_modarith_count++;

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    g_decrypt_time += elapsed.count();
    g_decrypt_count++;
}



//==============================================================================
//==============================================================================
//                             BENCHES AND TESTS
//                   FOR EXTRACTION AND ENCRYPTION/DECRYPTION
//==============================================================================
//==============================================================================


void Extract_Bench(const unsigned int nb_extr, MSK_Data * MSKD)
{
    clock_t t1, t2;
    float diff;
    unsigned int i;
    vec_ZZ id;
    ZZX SK_id[2];

    t1 = clock();

    cout << "0%" << flush;
    for(i=0; i<nb_extr; i++)
    {
        id = RandomVector();

        IBE_Extract(SK_id, id, MSKD);
        if((i+1)%(nb_extr/10)==0)
        {
            cout << "..." << (i+1)/(nb_extr/10) << "0%" << flush;
        }
    }

    t2 = clock();
    diff = ((float)t2 - (float)t1)/1000000.0F;
    cout << "\n\nIt took " << diff << " seconds to extract " << nb_extr << " keys." << endl;
    cout << "That's " << (diff/nb_extr)*1000 << " milliseconds per key." << endl << endl;
}


void Encrypt_Bench(const unsigned int nb_cryp, MPK_Data * MPKD, MSK_Data * MSKD)
{
    clock_t t1, t2;
    double diff;
    unsigned int i,j;
    vec_ZZ id;
    ZZX SK_id[2], w;
    CC_t SKid_FFT[N0];
    long int message[N0], decrypted[N0];
    long int identity[N0], Ciphertext[2][N0];


    id = RandomVector();
    IBE_Extract(SK_id, id, MSKD);
    IBE_Verify_Key(SK_id, id, MSKD);
    FFT_Interface_ZZXToFFT(SKid_FFT, SK_id[1]);
    for(i=0; i<N0; i++)
    {
        identity[i] = conv<long int>(id[i]);
    }

    t1 = clock();

    cout << "0%" << flush;
    for(i=0; i<nb_cryp; i++)
    {

        for(j=0; j<N0; j++)
        {
            message[j] = (rand()%2);
        }

        IBE_Encrypt(Ciphertext, message, identity, MPKD);
        IBE_Decrypt(decrypted, Ciphertext, SKid_FFT);

        if((i+1)%(nb_cryp/10)==0)
        {
            cout << "..." << (i+1)/(nb_cryp/10) << "0%" << flush;
        }
    }

    t2 = clock();
    diff = ((double)t2 - (double)t1)/1000000.0l;
    cout << "\n\nIt took " << diff << " seconds to do " << nb_cryp << " encryptions and decryptions." << endl;
    cout << "That's " << (diff/nb_cryp)*1000 << " milliseconds per encryption+decryption." << endl;
    cout << "That's " << (diff/nb_cryp)*1000*1024/N0 << " milliseconds per encryption+decryption per Kilobit." << endl << endl;
}


void Extract_Test(const unsigned int nb_extr, MSK_Data * MSKD)
{
    unsigned int i, rep;
    vec_ZZ id;
    ZZX SK_id[2];

    rep = 0;

    cout << "0%" << flush;
    for(i=0; i<nb_extr; i++)
    {
        id = RandomVector();

        IBE_Extract(SK_id, id, MSKD);
        rep += IBE_Verify_Key(SK_id, id, MSKD);
        if((i+1)%(nb_extr/10)==0)
        {
            cout << "..." << (i+1)/(nb_extr/10) << "0%" << flush;
        }
    }

    cout << endl;
    if(rep == 0)
    {    cout << endl << nb_extr << " extractions successfully performed!" << endl << endl;    }
    else
    {    cout << endl << rep << " out of " << nb_extr << " extractions failed miserabily!" << endl << endl;    }
}


void Encrypt_Test(const unsigned int nb_cryp, MPK_Data * MPKD, MSK_Data * MSKD)
{
    unsigned int i, j, rep;
    vec_ZZ id;
    ZZX SK_id[2], m, w;
    CC_t SKid_FFT[N0];
    long int id0[N0], Ciphertext[2][N0];
    long int message[N0], decrypted[N0];


    id = RandomVector();
    IBE_Extract(SK_id, id, MSKD);
    IBE_Verify_Key(SK_id, id, MSKD);
    FFT_Interface_ZZXToFFT(SKid_FFT, SK_id[1]);

    rep = 0;

    for(i=0; i<N0; i++)
    {
        id0[i] = conv<long int>(id[i]);
    }

    cout << "0%" << flush;
    for(i=0; i<nb_cryp; i++)
    {

        for(j=0; j<N0; j++)
        {
            message[j] = (rand()%2);
        }

        IBE_Encrypt(Ciphertext, message, id0, MPKD);
        IBE_Decrypt(decrypted, Ciphertext, SKid_FFT);
        
        for(j=0; j<N0; j++)
        {
            if(message[j] != decrypted[j])
            {
                cout << "ERROR : Dec(Enc(m)) != m " << endl;
                rep++;
                break;
            }
        }

        if((i+1)%(nb_cryp/10)==0)
        {
            cout << "..." << (i+1)/(nb_cryp/10) << "0%" << flush;
        }
    }

    cout << endl;
    if(rep == 0)
    {    cout << endl << nb_cryp << " encryptions+decryptions successfully performed!" << endl << endl;    }
    else
    {    cout << endl << rep << " out of " << nb_cryp << " encryptions+decryptions failed miserabily!" << endl << endl;    }
}
