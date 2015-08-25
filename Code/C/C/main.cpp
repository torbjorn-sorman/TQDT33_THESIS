#include <stdio.h>
#include <Windows.h>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <string.h>

#include "kiss_fft.h"

#include "tb_math.h"
#include "tb_image.h"
#include "tb_fft.h"
#include "tb_test.h"
#include "tb_transpose.h"

#define N 512 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

typedef kiss_fft_cpx tb_cpx; // tb_cpx

// 0.1255


int main()
{   
    double time;
    unsigned char result;
    result = test_equal_dft(tb_fft, kiss_fft, N, 0);

    if (result == 0)
        printf("Error: not equal...\n");
    
    time = test_time_dft(tb_fft, N);
    printf("TB I/O FFT:\t%fms\n", time);
    
    time = test_time_dft(kiss_fft, N);
    printf("KISS I/O FFT:\t%fms\n", time);
    
    result = test_equal_dft2d(tb_fft, kiss_fft, N, 0);

    if (result == 0)
        printf("Error 2D: not equal...\n");
    else
        printf("Success 2D!\n");

    /*
    time = test_time_dft_2d(tb_fft, N);
    printf("TB I/O FFT 2D:\t%fms\n", time);

    time = test_time_dft_2d(kiss_fft, N);
    printf("KISS IO FFT 2D:\t%fms\n", time);
    */
    //test_image(kiss_fft, "lena", N);
        
    printf("\n... done!\n");    
    getchar();
    return 0;
}