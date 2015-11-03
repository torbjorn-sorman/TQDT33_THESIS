#pragma once
#ifndef MYHELPEROPENCL_H
#define MYHELPEROPENCL_H

#include <CL\cl.h>

#include "../../Definitions.h"
#include "../../Common/mycomplex.h"
#include "../../Common/imglib.h"

#define OCL_GROUP_SIZE 512
#define OCL_TILE_DIM 32 // This is 8K local/shared mem
#define OCL_BLOCK_DIM 16 // This is 256 Work Items / Group

struct oclArgs {
    int n;
    int n_per_block;
    float dir;
    cl_uint workDim;
    size_t shared_mem_size;
    size_t data_mem_size;
    size_t global_work_size[3];
    size_t local_work_size[3];
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    cl_mem input, output;
    cl_platform_id platform;
    char *kernel_strings[2];
};

static void __inline swap(cl_mem *a, cl_mem *b)
{
    cl_mem c = *a;
    *a = *b;
    *b = c;
}

static void __inline ocl_set_kernel_args_global(oclArgs *args, cl_mem in, const float global_angle, unsigned int lmask, int steps, int dist)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(args->kernel, 1, sizeof(float), &global_angle);
    clSetKernelArg(args->kernel, 2, sizeof(unsigned int), &lmask);
    clSetKernelArg(args->kernel, 3, sizeof(int), &steps);
    clSetKernelArg(args->kernel, 4, sizeof(int), &dist);
}

static void __inline ocl_set_kernel_args_local(oclArgs *args, cl_mem in, cl_mem out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range_half)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(args->kernel, 2, args->shared_mem_size, NULL);
    clSetKernelArg(args->kernel, 3, sizeof(float), &local_angle);
    clSetKernelArg(args->kernel, 4, sizeof(int), &steps_left);
    clSetKernelArg(args->kernel, 5, sizeof(int), &leading_bits);
    clSetKernelArg(args->kernel, 6, sizeof(float), &scalar);
    clSetKernelArg(args->kernel, 7, sizeof(int), &block_range_half);
}

static void __inline oclSetKernelGPU2DArg(oclArgs *args, cl_mem in, cl_mem out, float local_angle, int steps_left, int leading_bits, float scalar, int n_per_block)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(args->kernel, 2, args->shared_mem_size, NULL);
    clSetKernelArg(args->kernel, 3, sizeof(float), &local_angle);
    clSetKernelArg(args->kernel, 4, sizeof(int), &steps_left);
    clSetKernelArg(args->kernel, 5, sizeof(int), &leading_bits);
    clSetKernelArg(args->kernel, 6, sizeof(float), &scalar);
    clSetKernelArg(args->kernel, 7, sizeof(int), &n_per_block);
}

static void __inline oclSetKernelTransposeArg(oclArgs *args, cl_mem in, cl_mem out)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(args->kernel, 2, args->shared_mem_size, NULL);
    clSetKernelArg(args->kernel, 3, sizeof(int), &args->n);
}

cl_int checkErr(cl_int error, char *msg);
std::string getKernel(const char *filename);
cl_int ocl_create_kernels(oclArgs *arg_cpu, oclArgs *arg_gpu, cpx *data_in, transform_direction dir, const int n);
cl_int ocl_create_timestamp_kernel(oclArgs *arg_target, oclArgs *arg_tm);
cl_int oclCreateKernels2D(oclArgs *arg_cpu, oclArgs *arg_gpu, oclArgs *arg_transpose, cpx *data_in, transform_direction dir, const int n);
cl_int oclRelease(cpx *dev_in, cpx *dev_out, oclArgs *arg_cpu, oclArgs *arg_gpu);
cl_int oclRelease2D(cpx *dev_in, cpx *dev_out, oclArgs *arg_cpu, oclArgs *arg_gpu, oclArgs *arg_transpose);
int freeResults(cpx **din, cpx **dout, cpx **dref, const int n);
void setupBuffers(cpx **in, cpx **out, cpx **ref, const int n);

#endif