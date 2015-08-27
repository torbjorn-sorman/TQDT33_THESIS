#include <stdio.h>

#include "tb_definitions.h"

#include "tb_math.h"
#include "tb_image.h"
#include "tb_fft.h"
#include "tb_test.h"
#include "tb_transpose.h"
#include "tb_print.h"

//#define n 65536 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

int main()
{
    double time;
    fft_function fn, ref;
    fn = tb_fft_openmp;
    //fn = tb_fft_single;
    ref = tb_fft;
    //simple();
    test_equal_dft(fn, ref, 1);
    console_separator(1);

    time = test_cmp_time(fn, ref);
    printf("Avg. time diff: %f%%\n", time);
    console_separator(1);

    test_equal_dft2d(fn, ref, 1);
    console_separator(1);
        
    time = test_time_2d(0);
    printf("TB FFT 2D:\t%fms\n", time);
    console_separator(1);

    time = test_time_2d(1);
    printf("TB OMP FFT 2D:\t%fms\n", time);
    console_separator(1);
    //test_image(tb_fft, "lena", n);
    //*/
    printf("\n... done!\n");
    getchar();
    return 0;
}