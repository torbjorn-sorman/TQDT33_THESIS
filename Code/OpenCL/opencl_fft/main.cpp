#include <stdio.h>
#include "fftGPUSync.h"

int main(int argc, char** argv)
{
    for (int i = 4; i < 4096; i *= 2)
        GPUSync_validate(i);
    printf("\nCompleted\n");
    getchar();
    return 0;
}