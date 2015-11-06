#pragma once
#ifndef MYHELPEROPENCL_H
#define MYHELPEROPENCL_H

#include <CL\cl.h>

#include "../../Definitions.h"
#include "../../Common/mycomplex.h"
#include "../../Common/imglib.h"

struct ocl_args {
    int n;
    int n_per_block;
    int number_of_blocks;
    float dir;
    cl_uint workDim;
    size_t shared_mem_size;
    size_t data_mem_size;
    size_t work_size[3];
    size_t group_work_size[3];
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    cl_mem input, output;
    cl_platform_id platform;
};

static void __inline swap(cl_mem *a, cl_mem *b)
{
    cl_mem c = *a;
    *a = *b;
    *b = c;
}

static void __inline ocl_set_args(ocl_args *args, cl_mem in, const float global_angle, unsigned int lmask, int steps, int dist)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(args->kernel, 1, sizeof(float), &global_angle);
    clSetKernelArg(args->kernel, 2, sizeof(unsigned int), &lmask);
    clSetKernelArg(args->kernel, 3, sizeof(int), &steps);
    clSetKernelArg(args->kernel, 4, sizeof(int), &dist);
}

static void __inline ocl_set_args(ocl_args *args, cl_mem in, cl_mem out, float local_angle, int steps_left, int leading_bits, float scalar, int block_range)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(args->kernel, 2, args->shared_mem_size, NULL);
    clSetKernelArg(args->kernel, 3, sizeof(float), &local_angle);
    clSetKernelArg(args->kernel, 4, sizeof(int), &steps_left);
    clSetKernelArg(args->kernel, 5, sizeof(int), &leading_bits);
    clSetKernelArg(args->kernel, 6, sizeof(float), &scalar);
    clSetKernelArg(args->kernel, 7, sizeof(int), &block_range);
}

static void __inline ocl_set_args(ocl_args *args, cl_mem in, cl_mem out)
{
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(args->kernel, 2, args->shared_mem_size, NULL);
    clSetKernelArg(args->kernel, 3, sizeof(int), &args->n);
}

cl_int ocl_check_err(cl_int error, char *msg);
cl_int ocl_setup(ocl_args *a_host, ocl_args *a_dev, cpx *data_in, transform_direction dir, const int group_size, const int n);
cl_int ocl_setup_timestamp(ocl_args *arg_target, ocl_args *arg_tm);
cl_int ocl_setup(ocl_args *a_host, ocl_args *a_dev, ocl_args *a_trans, cpx *data_in, transform_direction dir, const int group_size, const int tile_dim, const int block_dim, const int n);
cl_int ocl_shakedown(cpx *dev_in, cpx *dev_out, ocl_args *a_host, ocl_args *a_dev);
cl_int ocl_shakedown(cpx *dev_in, cpx *dev_out, ocl_args *a_host, ocl_args *a_dev, ocl_args *a_trans);
int ocl_free(cpx **din, cpx **dout, cpx **dref, const int n);
void ocl_setup_buffers(cpx **in, cpx **out, cpx **ref, const int n);

#endif