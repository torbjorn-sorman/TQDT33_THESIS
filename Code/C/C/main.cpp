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

#define N 8192 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

typedef kiss_fft_cpx tb_cpx; // tb_cpx

// 0.1255


int main()
{   
    LARGE_INTEGER freq, tStart, tStop;
    /*
    const int tests = 2;
    
    tb_cpx **in = (tb_cpx **)malloc(sizeof(tb_cpx) * N * N);
    tb_cpx **in2 = (tb_cpx **)malloc(sizeof(tb_cpx) * N * N);
    for (uint32_t y = 0; y < N; ++y) {
        in[y] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        in2[y] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        for (uint32_t x = 0; x < N; ++x) { 
            in[y][x] = { (float)x, (float)y };
            in2[y][x] = { (float)x, (float)y }; 
        }
    }
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&tStart);
    for (int i = 0; i < tests; ++i){ 
        transpose(in, N); 
    } 
    QueryPerformanceCounter(&tStop);
    printf("Time naive:\t%.2fms\n\n", (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart);

    const int cut = 2 * 2;
    for (int j = cut; j < N / cut; j *= 2) { QueryPerformanceCounter(&tStart); for (int i = 0; i < tests; ++i) { transpose_block(in2, N, j); } 
    QueryPerformanceCounter(&tStop); printf("Time block %d:\t%.2fms\n", j, (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart); }
    for (uint32_t y = 0; y < N; ++y) {
        free(in[y]);
        free(in2[y]);
    }
    free(in);
    free(in2);
    */
    /*
    QueryPerformanceFrequency(&freq);
    int runs = 8192;
    int c = 0;
    QueryPerformanceCounter(&tStart);
    for (int i = 0; i < runs; ++i){
        int n = 512 + runs;
        for (int i = 0; i < n; ++i){
            ++c;
        }
    }
    QueryPerformanceCounter(&tStop);
    printf("Time outer:\t%.2fms\n\n", (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart);

    QueryPerformanceCounter(&tStart);
    for (int i = 0; i < runs; ++i){
        for (int i = 0; i < 512 + runs; ++i){
            ++c;
        }
    }
    QueryPerformanceCounter(&tStop);
    printf("Time inner:\t%.2fms\n\n", (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart);
    */
    double time;
    unsigned char result;
    result = test_equal_dft(tb_fft, kiss_fft, N, 0);

    if (result == 0)
        printf("Error: not equal...\n");
    
    time = test_time_dft(tb_fft, N);
    printf("TB I/O FFT:\t%fms\n", time);
    
    time = test_time_dft(kiss_fft, N);
    printf("KISS I/O FFT:\t%fms\n", time);
    
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