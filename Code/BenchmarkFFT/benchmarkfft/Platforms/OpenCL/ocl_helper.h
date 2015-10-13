#ifndef MYHELPEROPENCL_H
#define MYHELPEROPENCL_H

#include <iostream>
#include <vector>
#include <fstream>
#include <CL\cl.h>

#include "../../Definitions.h"
#include "../../Common/mycomplex.h"
#include "../../Common/imglib.h"

struct oclArgs {
    int n;
    int nBlock;
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
    cl_mem sync_in, sync_out;
    cl_platform_id platform;
    char *kernelSource;
};

cl_int checkErr(cl_int error, char *msg);
std::string getKernel(const char *filename);
cl_int oclCreateKernels(oclArgs *argCPU, oclArgs *argGPU, cpx *data_in, fftDir dir, const int n);
cl_int oclCreateKernels2D(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans, cpx *data_in, fftDir dir, const int n);
cl_int oclRelease(cpx *dev_out, oclArgs *argCPU, oclArgs *argGPU);
cl_int oclRelease2D(cpx *dev_in, cpx *dev_out, oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans);
int freeResults(cpx **din, cpx **dout, cpx **dref, const int n);
void setupBuffers(cpx **in, cpx **out, cpx **ref, const int n);

static cl_int __inline oclExecute(oclArgs *args)
{    
    cl_int err = clEnqueueNDRangeKernel(args->commands, args->kernel, args->workDim, NULL, args->global_work_size, args->local_work_size, 0, NULL, NULL);
    if (err) return err;
    return clFinish(args->commands);
}

static void __inline swap(cl_mem *a, cl_mem *b)
{
    cl_mem c = *a;
    *a = *b;
    *b = c;
}

static void __inline oclSetKernelCPUArg(oclArgs *args, cl_mem in, cl_mem out, float w_angle, unsigned int lmask, int steps, int dist)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(args->kernel, 2, sizeof(float), &w_angle);
    clSetKernelArg(args->kernel, 3, sizeof(unsigned int), &lmask);
    clSetKernelArg(args->kernel, 4, sizeof(int), &steps);
    clSetKernelArg(args->kernel, 5, sizeof(int), &dist);
}

static void __inline oclSetKernelGPUArg(oclArgs *args, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, int n2)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &args->input);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &args->output);
    clSetKernelArg(args->kernel, 2, sizeof(cl_mem), &args->sync_in);
    clSetKernelArg(args->kernel, 3, sizeof(cl_mem), &args->sync_out);
    clSetKernelArg(args->kernel, 4, args->shared_mem_size, NULL);
    clSetKernelArg(args->kernel, 5, sizeof(float), &angle);
    clSetKernelArg(args->kernel, 6, sizeof(float), &bAngle);
    clSetKernelArg(args->kernel, 7, sizeof(int), &depth);
    clSetKernelArg(args->kernel, 8, sizeof(int), &lead);
    clSetKernelArg(args->kernel, 9, sizeof(int), &breakSize);
    clSetKernelArg(args->kernel, 10, sizeof(cpx), &scale);
    clSetKernelArg(args->kernel, 11, sizeof(int), &nBlocks);
    clSetKernelArg(args->kernel, 12, sizeof(int), &n2);
}

static void __inline oclSetKernelGPU2DArg(oclArgs *args, cl_mem in, cl_mem out, float bAngle, int depth, cpx scale, int nBlocks2)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &args->input);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &args->output);
    clSetKernelArg(args->kernel, 2, args->shared_mem_size, NULL);
    clSetKernelArg(args->kernel, 3, sizeof(float), &bAngle);
    clSetKernelArg(args->kernel, 4, sizeof(int), &depth);
    clSetKernelArg(args->kernel, 5, sizeof(cpx), &scale);
    clSetKernelArg(args->kernel, 6, sizeof(int), &nBlocks2);
}

static void __inline oclSetKernelTransposeArg(oclArgs *args, cl_mem in, cl_mem out)
{
    checkErr(clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &in), "input");
    checkErr(clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &out), "output");
    checkErr(clSetKernelArg(args->kernel, 2, args->shared_mem_size, NULL), "shmem");
    checkErr(clSetKernelArg(args->kernel, 3, sizeof(int), &args->n), "n");
}

#endif