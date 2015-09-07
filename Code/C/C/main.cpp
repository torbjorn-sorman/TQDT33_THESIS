#include <stdio.h>

#include "tb_definitions.h"

#include "tb_math.h"
#include "tb_image.h"
#include "tb_fft.h"
#include "tb_test.h"
#include "tb_transpose.h"
#include "tb_print.h"
#include "tb_fft_helper.h"
#include "test_seq.h"

#include "fft_reg.h"
#include "fft_reg_omp.h"
#include "fft_const_geom.h"
#include "fft_tobb.h"

#include <omp.h> 

//#define n 65536 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

void printBin(unsigned int val)
{
    for (int i = 31; i >= 0; --i) {
        printf("%u", (val >> i) & 1);
    }
    printf(" %u\n", val);
}

void genBitmask(char *name)
{
    char filename[64] = "";
    FILE *f;
    strcat_s(filename, "out/");
    strcat_s(filename, name);
    strcat_s(filename, ".txt");
    fopen_s(&f, filename, "w");
    printf("Generate bitmask...\n");
    for (int i = 1; i <= 32; ++i) {
        unsigned int val = 0;
        for (int n = i; n <= 32; n += i) {
            val |= 1 << (n - 1);
        }
        printBin(val);
        fprintf_s(f, "%u,\n", val);
    }
    printf("Filename: %s\n", filename);
    fclose(f);
}

unsigned int power(unsigned int base, int exp)
{
    if (exp == 0)
        return 1;
    unsigned int value = base;
    for (int i = 0; i < exp; ++i) {
        value *= base;
    }
    return value;
}

unsigned int power2(int exp)
{
    return power(2, exp);
}

int main()
{    
    const unsigned int n = power2(19);
    int n_threads;
    double time;
    time = 0.0;

#ifdef _OPENMP
    omp_set_num_threads(4);
    n_threads = omp_get_max_threads();
#else
    n_threads = 1;
#endif

    //genBitmask("bitmask");

    //test_omp();
    //printf("time: %f\n", timing(twiddle_factors, 1, 8192));
    test_fft("Reg FFT", fft_reg, n_threads, 1, n);
    test_fft("Tobb FFT", fft_tobb, n_threads, 1, n);
    test_fft("Constant Geometry FFT", fft_const_geom, n_threads, 1, n);
    //test_fft("Const Geom FFT", fft_const_geom, n_threads);
    
    //printf("\n... done!\n");
    getchar();
    return 0;
}