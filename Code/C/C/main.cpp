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

#define N 16 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

typedef kiss_fft_cpx tb_cpx; // tb_cpx

// 0.1255

void simple()
{
    uint32_t i;
    tb_cpx *in = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    tb_cpx *in2 = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    tb_cpx *out = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    tb_cpx *out2 = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    for (i = 0; i < N; ++i) {
        in[i].r = 0.f;
        in[i].i = 0.f;
        in2[i].r = 0.f;
        in2[i].i = 0.f;
    }
    in[1].r = 1;
    in2[1].r = 1;
    tb_fft(FORWARD_FFT, in, out, N);
    kiss_fft(FORWARD_FFT, in2, out2, N);
    for (i = 0; i < N; ++i) {
        printf("(%f, %f)\t(%f, %f)\n", out[i].r, out[i].i, out2[i].r, out2[i].i);
    }
    printf("\n");
    tb_fft(INVERSE_FFT, out, in, N);
    kiss_fft(INVERSE_FFT, out2, in2, N);
    for (i = 0; i < N; ++i) {
        printf("(%f, %f)\t(%f, %f)\n", in[i].r, in[i].i, in2[i].r, in2[i].i);
    }
}

int main()
{   
    simple();
    getchar();
    return 0;
    double time;
    unsigned char result;
    result = test_equal_dft(tb_fft, kiss_fft, N, 1);
    result = test_transpose(N);
        
    time = test_time_dft(tb_fft, N);
    printf("TB I/O FFT:\t%fms\n", time);
    
    time = test_time_dft(kiss_fft, N);
    printf("KISS I/O FFT:\t%fms\n\n", time);
    
    /*
    result = test_equal_dft2d(tb_fft, kiss_fft, N, 0);

    if (result == 0)
        printf("Error 2D: not equal...\n");
    else
        printf("Success 2D!\n");
        */
    /*
    time = test_time_dft_2d(tb_fft, N);
    printf("TB I/O FFT 2D:\t%fms\n", time);

    time = test_time_dft_2d(kiss_fft, N);
    printf("KISS IO FFT 2D:\t%fms\n", time);
    */
    test_image(kiss_fft, "lena", N);
        
    printf("\n... done!\n");    
    getchar();
    return 0;
}