#include <stdio.h>

#include "tb_definitions.h"

#include "tb_math.h"
#include "tb_image.h"
#include "tb_fft.h"
#include "tb_test.h"
#include "tb_transpose.h"
#include "tb_print.h"
#include "tb_fft_helper.h"
#include "test_seq.h"
#include "genCode.h"

#include "fft_reg.h"
#include "fft_const_geom.h"
#include "fft_tobb.h"
#include "fft_fixed.h"

#include <omp.h> 

//#define n 65536 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

unsigned int power(unsigned int base, int exp)
{
    if (exp == 0)
        return 1;
    unsigned int value = base;
    for (int i = 1; i < exp; ++i) {
        value *= base;
    }
    return value;
}

unsigned int power2(int exp)
{
    return power(2, exp);
}



int main()
{
    int n_threads;
#ifdef _OPENMP
    omp_set_num_threads(4);
    n_threads = omp_get_max_threads();
#else
    n_threads = 1;
#endif

    const unsigned int n = power2(3);

    //test_image(fft2d_tobb, "crevisio", n_threads, 4096);

    //test_fft2d("FFT Constant Geometry 2D", fft2d_const_geom, n_threads, 1, 4096);
    //test_fft2d("FFT Regular 2D", fft2d_reg, n_threads, 1, 4096);
    //test_fft2d("FFT Tobb 2D", fft2d_tobb, n_threads, 1, 4096);
//#define GEN_CODE
#ifdef GEN_CODE
    createFixedSizeFFT("const", n);
#else
    //test_fftw(n);    
    //test_fft("Reg FFT", fft_reg, n_threads, 1, n);
    //test_fft("Tobb FFT", fft_tobb, n_threads, 1, n);
    test_fft("Constant Geometry FFT", fft_const_geom, n_threads, 1, n);
    test_fft("Fixed FFT", fft_fixed, n_threads, 1, n);

    //test_short_fft(fft_fixed, n_threads, n);
    //test_short_fft(fft_const_geom, n_threads, n);
#endif

    printf("\nComplete!\n");
    getchar();
    return 0;
}