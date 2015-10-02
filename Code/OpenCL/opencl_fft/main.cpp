#include <stdio.h>
#include "fftOpenCL.h"
#include "helper.h"

#define START 4
#define END 65536

int main(int argc, char** argv)
{
    //double measurements[64];
    int index = 0;

    for (int i = START; i <= END; i *= 2)
        runPartialSync(i);
    /*
    for (int i = START; i <= END; i *= 2)
        measurements[index++] = GPUSync_performance(i);
        
    for (int i = START; i <= END; i *= 2)
        printf("%d:\t%f\n", i, measurements[log2_32(i) - log2_32(START)]);
    */
    printf("\nCompleted\n");
    getchar();
    return 0;
}