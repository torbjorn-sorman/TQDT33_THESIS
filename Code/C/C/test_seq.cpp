#ifdef _OPENMP
#include <omp.h> 
#endif

#include "test_seq.h"

#include "tb_fft.h"
#include "tb_fft_helper.h"
#include "tb_test.h"
#include "tb_print.h"
#include "tb_test.h"

void test_seq_fft(const int n_threads)
{
#ifdef _OPENMP  
    printf("Running on max threads: %d\n", omp_get_max_threads());
#else
    printf("Running on single thread/core.\n");
#endif;
    //test_complete_fft("REGULAR", fft_body, n_threads);
    //test_complete_fft_cg("CONST GEOM", n_threads);
    //test_complete_fft_cg_no_twiddle("CONST GEOM NO TWIDDLE", n_threads);
    //test_complete_ext("CPG FFT", cgp_fft, n_threads);
}

void test_seq_twiddle(const int n_threads, int size)
{
    double time;
    time = 0.0;
    /*
    twiddle_fn tw_ref = twiddle_factors_s;

    test_twiddle(twiddle_factors_alt, tw_ref, n_threads, size);
    time = test_time_twiddle(twiddle_factors_alt, n_threads, size);
    printf("Time twiddle_factors_alt:\t%.1f\n", time);

    test_twiddle(twiddle_factors, tw_ref, n_threads, size);
    time = test_time_twiddle(twiddle_factors, n_threads, size);
    printf("Time twiddle_factors:\t\t%.1f\n", time);

    test_twiddle(twiddle_factors_s, tw_ref, n_threads, size);
    time = test_time_twiddle(twiddle_factors_s, n_threads, size);
    printf("Time twiddle_factors_s:\t\t%.1f\n", time);
    */
}