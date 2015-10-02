#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "CL\cl.h"

typedef const float fftDir;

struct cpx {
    float x;
    float y;
};

struct oclArgs {
    int n;
    int nBlock;
    float dir;
    size_t shared_mem_size;
    size_t data_mem_size;
    size_t global_work_size;
    size_t local_work_size;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    cl_mem input, output, sync_in, sync_out;
    cl_platform_id platform;
    char *kernelSource;
};

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FFT_FORWARD -1.0f
#define FFT_INVERSE 1.0f

#define SHARED_MEM_SIZE 49152   // 48K assume: 49152 bytes. Total mem size is 65536, where 16384 is cache if 
#define MAX_BLOCK_SIZE 1024     // this limits tsCombineGPUSync!
#define TILE_DIM 64
#define THREAD_TILE_DIM 32      // This squared is the number of threads per block in the transpose kernel.

#endif