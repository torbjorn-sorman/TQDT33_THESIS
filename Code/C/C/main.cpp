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

#define N 1024 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

typedef kiss_fft_cpx my_complex;

void transpose(my_complex **seq, uint32_t n)
{
    uint32_t x, y;
    my_complex tmp;
    for (y = 0; y < n; ++y) {
        for (x = y + 1; x < n; ++x) {
            tmp = seq[y][x];
            seq[y][x] = seq[x][y];
            seq[x][y] = tmp;
        }
    }
}

/*  OBSERVE size % block_size = 0
    size > 1024, block_size seems fastest at 4 else 16...
*/
void transpose_block(my_complex **seq, const uint32_t size, const uint32_t block_size)
{
    uint32_t blx, bly, x, y;
    my_complex tmp;
    for (bly = 0; bly < size; bly += block_size) {
        for (blx = bly; blx < size; blx += block_size) {
            for (y = bly; y < block_size + bly; ++y) {
                for (x = blx; x < block_size + blx; ++x) {
                    if (x > y) {
                        tmp = seq[y][x];
                        seq[y][x] = seq[x][y];
                        seq[x][y] = tmp;
                    }
                }
            }
        }
    }
}

int main()
{   
    const int tests = 2;
    LARGE_INTEGER freq, tStart, tStop;
    my_complex **in = (my_complex **)malloc(sizeof(my_complex) * N * N);
    my_complex **in2 = (my_complex **)malloc(sizeof(my_complex) * N * N);
    for (uint32_t y = 0; y < N; ++y) {
        in[y] = (my_complex *)malloc(sizeof(my_complex) * N);
        in2[y] = (my_complex *)malloc(sizeof(my_complex) * N);
        for (uint32_t x = 0; x < N; ++x) { 
            in[y][x] = { (float)x, (float)y };
            in2[y][x] = { (float)x, (float)y }; 
        }
    }
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&tStart); for (int i = 0; i < tests; ++i){ transpose(in, N); } QueryPerformanceCounter(&tStop);
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
    /*
    double time;
    unsigned char result;
    result = test_equal_dft(tb_fft, kiss_fft, N, 0);

    if (result == 0)
        printf("Error: not equal...\n");
    
    time = test_time_dft(tb_fft, N);
    printf("TB I/O FFT:\t%fms\n", time);
    
    time = test_time_dft(tb_fft_test, N);
    printf("KISS I/O FFT:\t%fms\n", time);
    */
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