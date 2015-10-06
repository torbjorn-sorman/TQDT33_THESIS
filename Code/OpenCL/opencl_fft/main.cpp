#include <stdio.h>
#include "fftOpenCL.h"
#include "helper.h"

#define START 4
#define END 33554432


//#define PROFILER

int main(int argc, char** argv)
{
    double measurements[64];
    int index = 0;

#ifndef PROFILER
    printf("Performance...");
#endif
    for (int i = START; i <= END; i *= 2) {
        measurements[index++] = GPUSync_performance(i);
#ifndef PROFILER
        if (GPUSync_validate(i))
            printf("!%d!", i);
        else
            printf(".");
#endif
    }
#ifndef PROFILER
    printf("\n");
    char *fmt;
    for (int i = START; i <= END; i *= 2) {
        fmt = (i < 1000000 ? "%d:\t\t%.0f\n" : "%d:\t%.0f\n");  
        printf(fmt, i, measurements[log2_32(i) - log2_32(START)]);
    }

    printf("\nCompleted\n");
    getchar();
#endif
    return 0;
}