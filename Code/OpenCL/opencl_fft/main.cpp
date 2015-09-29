#include <stdio.h>
#include "fftGPUSync.h"

int main(int argc, char** argv)
{
    GPUSync_validate(8);
    printf("\nCompleted\n");
    getchar();
    return 0;
}