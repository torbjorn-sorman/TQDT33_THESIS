#ifdef _OPENMP
#include <omp.h> 
#endif

#include "test_seq.h"

#include "tb_fft.h"
#include "tb_fft_helper.h"
#include "tb_test.h"
#include "tb_print.h"
#include "tb_test.h"

void test_seq_fft()
{
#ifdef _OPENMP  
    printf("Running on max threads: %d\n", omp_get_max_threads());
#else
    printf("Running on single thread/core.\n");
#endif;
    test_complete_fft("REGULAR", fft_body);
    test_complete_fft_cg("CONST GEOM");
    //test_complete_ext("CPG FFT", cgp_fft);
}

void test_seq_twiddle(int size)
{
    double time;
    time = 0.0;

    twiddle_fn tw_ref = twiddle_factors_s;

    test_twiddle(twiddle_factors_alt, tw_ref, size);
    time = test_time_twiddle(twiddle_factors_alt, size);
    printf("Time twiddle_factors_alt:\t%.1f\n", time);

    test_twiddle(twiddle_factors, tw_ref, size);
    time = test_time_twiddle(twiddle_factors, size);
    printf("Time twiddle_factors:\t\t%.1f\n", time);

    test_twiddle(twiddle_factors_s, tw_ref, size);
    time = test_time_twiddle(twiddle_factors_s, size);
    printf("Time twiddle_factors_s:\t\t%.1f\n", time);

}