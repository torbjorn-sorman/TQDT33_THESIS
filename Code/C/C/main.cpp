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

#include "fft_regular.h"
#include "fft_reg.h"
#include "fft_reg_omp.h"
#include "fft_const_geom.h"
#include "fft_tobb.h"

#ifdef _OPENMP
#include <omp.h> 
#endif
//#define n 65536 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

int main()
{
    const unsigned int n = 1048576;
    int n_threads;
    double time;
    time = 0.0;

#ifdef _OPENMP
    omp_set_num_threads(4);
    n_threads = omp_get_max_threads();
#else
    n_threads = 1;
#endif

    test_fft("Reg OMP FFT", fft_reg_omp, n_threads);
    test_fft("Constant Geometry FFT", fft_const_geom, n_threads);
    test_fft("Reg FFT", fft_reg, n_threads);
    //test_fft("Tobb FFT", fft_tobb, n_threads);

    printf("\n... done!\n");
    getchar();
    return 0;
}