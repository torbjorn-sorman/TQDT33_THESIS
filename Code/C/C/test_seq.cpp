#include "test_seq.h"

#include "tb_fft_helper.h"
#include "tb_test.h"
#include "tb_print.h"

void test_seq_fft(int size)
{
    double time;
    time = 0.0;
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

    console_separator(1);

    test_twiddle(twiddle_factors_alt, tw_ref, size);
    time = test_time_twiddle(twiddle_factors_alt_omp, size);
    printf("Time twiddle_factors_alt_omp:\t%.1f\n", time);

    test_twiddle(twiddle_factors_omp, tw_ref, size);
    time = test_time_twiddle(twiddle_factors_omp, size);
    printf("Time twiddle_factors_omp:\t%.1f\n", time);

    test_twiddle(twiddle_factors_s_omp, tw_ref, size);
    time = test_time_twiddle(twiddle_factors_s_omp, size);
    printf("Time twiddle_factors_s_omp:\t%.1f\n", time);

    console_separator(1);

    time = test_time_reverse(bit_reverse, size);
    printf("Time bit_reverse:\t\t%.1f\n", time);

    time = test_time_reverse(bit_reverse_omp, size);
    printf("Time bit_reverse_omp:\t\t%.1f\n", time);
}