#include <stdio.h>
#include <Windows.h>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <string.h>

#include "tb_definitions.h"

#include "tb_math.h"
#include "tb_image.h"
#include "tb_fft.h"
#include "tb_test.h"
#include "tb_transpose.h"
#include "tb_print.h"

//#define N 65536 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

typedef kiss_fft_cpx tb_cpx; // tb_cpx

int main()
{   
    double time;
    test_equal_dft(tb_fft, tb_fft_old, 0);
    time = test_cmp_time(tb_fft, tb_fft_old);
    
    printf("Avg. time diff: %f%%\n", time);
    //uint32_t n;
    //unsigned char result;
    //result = test_equal_dft(tb_fft, kiss_fft, 1);
    //result = test_transpose(N);
    /*
    test_equal_dft2d(tb_fft, kiss_fft, 0);

    time = test_time_dft_2d(tb_fft, N);
    printf("TB I/O FFT 2D:\t%fms\n", time);

    time = test_time_dft_2d(kiss_fft, N);
    printf("KISS IO FFT 2D:\t%fms\n", time);
    */
    //test_image(tb_fft, "lena", N);
        
    printf("\n... done!\n");    
    getchar();
    return 0;
}