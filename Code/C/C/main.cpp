#include <stdio.h>

#include "tb_definitions.h"

#include "tb_math.h"
#include "tb_image.h"
#include "tb_fft.h"
#include "tb_test.h"
#include "tb_transpose.h"
#include "tb_print.h"
#include "tb_fft_helper.h"

//#define n 65536 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

int main()
{
    int size;
    double time;
    size = 512;
    time = test_time_dft_2d(tb_fft2d, tb_fft, size);
    /*
    printf("Eq: %d\n", test_twiddle(twiddle_factors_omp, twiddle_factors, size));

    time = test_time_twiddle(twiddle_factors, size);
    printf("Twiddle Smart:\t%fms\n", time);
    console_separator(1);

    time = test_time_twiddle(twiddle_factors_omp, size);
    printf("Twiddle OMP:\t%fms\n", time);
    console_separator(1);
    */
    //printf("FFT time: %f\n", time);
    /*
    test_equal_dft(tb_fft, tb_fft_openmp, 1);
    console_separator(1);
    */
    /*
    time = test_cmp_time(tb_fft_openmp, tb_fft);
    printf("Avg. time diff: %f%%\n", time);
    console_separator(1);
    */
    /*
    test_equal_dft2d(tb_fft2d, tb_fft_openmp, tb_fft, 1);
    console_separator(1);
      */
    /*
    time = test_time_dft_2d(tb_fft2d, tb_fft, size);
    printf("TB FFT 2D:\t%fms\n", time);
    console_separator(1);
    
    time = test_time_dft_2d(tb_fft2d_openmp_alt, tb_fft, size);
    printf("TB OMP FFT 2D:\t%fms\n", time);
    console_separator(1);
    */
    //test_image(tb_fft2d, tb_fft, "lena", size);
    /*
    generate_test_image_set("img/photo05/4096.ppm", "img/photo05/", 4096);
    generate_test_image_set("img/lena/4096.ppm", "img/lena/", 4096);
    generate_test_image_set("img/crevisio/4096.ppm", "img/crevisio/", 4096);
    */
    //printf("\n... done!\n");
    //getchar();
    return 0;
}