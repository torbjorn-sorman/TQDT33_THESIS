#include <stdio.h>

#include "tb_definitions.h"

#include "tb_math.h"
#include "tb_image.h"
#include "tb_fft.h"
#include "tb_test.h"
#include "tb_transpose.h"
#include "tb_print.h"
#include "tb_fft_helper.h"

#include <omp.h>
//#define n 65536 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

int main()
{
    int size;
    double time;
    size = 512;

    test_image(tb_fft2d, tb_fft, "photo05", size);
   
    printf("\n... done!\n");
    getchar();
    return 0;
}