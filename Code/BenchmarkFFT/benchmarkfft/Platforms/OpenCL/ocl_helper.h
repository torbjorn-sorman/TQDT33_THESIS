#ifndef MYHELPEROPENCL_H
#define MYHELPEROPENCL_H

#include <vector>
#include <fstream>
#include <CL\cl.h>

#include "../../Definitions.h"
#include "../../Common/mycomplex.h"

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

static void __inline oclExecute(oclArgs *args)
{
    clEnqueueNDRangeKernel(args->commands, args->kernel, 1, NULL, &args->global_work_size, &args->local_work_size, 0, NULL, NULL);
    clFinish(args->commands);
}

static void __inline swap(cl_mem *a, cl_mem *b)
{
    cl_mem c = *a;
    *a = *b;
    *b = c;
}

static void __inline oclSetKernelCPUArg(oclArgs *args, float w_angle, unsigned int lmask, int steps, int dist)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &args->input);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &args->output);
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

std::string getKernel(const char *filename);

int checkErr(cl_int error, char *msg);
int checkErr(cl_int error, cl_int args, char *msg);

cl_int oclSetup(char *kernelName, cpx *dev_in, oclArgs *args);
cl_int oclSetupKernel(const int n, oclArgs *args);
cl_int oclSetupProgram(char *kernelFilename, char *kernelName, oclArgs *args);
cl_int oclSetupDeviceMemoryData(oclArgs *args, cpx *dev_in);
cl_int oclSetupWorkGroupsAndMemory(oclArgs *args, oclArgs *argsCrossGroups);
cl_int oclCreateKernels(oclArgs *argCPU, oclArgs *argGPU, cpx *data_in, fftDir dir, const int n);
cl_int oclRelease(cpx *dev_out, oclArgs *argCPU, oclArgs *argGPU);
int freeResults(cpx **din, cpx **dout, cpx **dref, const int n);

#endif