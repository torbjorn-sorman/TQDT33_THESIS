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

#include <omp.h>
//#define n 65536 // 8192, 65536, 1048576, 2097152, 4194304, 8388608, 16777216

int main()
{
    const unsigned int size = 2147483648 / 2;
    double time;
    time = 0.0;

    test_seq_fft();
   
    printf("\n... done!\n");
    getchar();
    return 0;
}