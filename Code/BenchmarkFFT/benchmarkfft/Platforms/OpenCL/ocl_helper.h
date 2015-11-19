#pragma once
#ifndef MYHELPEROPENCL_H
#define MYHELPEROPENCL_H

//#if VENDOR_SELECTED == VENDOR_AMD
//#include <AMD/CL/cl.h>
//#elif VENDOR_SELECTED == VENDORE_NVIDIA
#include <CL/cl.h>
//#endif

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

cl_int ocl_check_err(cl_int error, char *msg);
cl_int ocl_setup(ocl_args *a_host, ocl_args *a_dev, cpx *data_in, transform_direction dir, const int group_size, const int n);
cl_int ocl_setup_timestamp(ocl_args *arg_target, ocl_args *arg_tm);
cl_int ocl_setup(ocl_args *a_host, ocl_args *a_dev, ocl_args *a_trans, cpx *data_in, transform_direction dir, const int group_size, const int tile_dim, const int block_dim, const int n);
cl_int ocl_setup_kernels(ocl_args *args, const int group_size, bool dim2);
cl_int ocl_setup_program(std::string kernel_filename, char *kernel_name, ocl_args *args);
cl_int ocl_setup_program(std::string kernel_filename, char *kernel_name, ocl_args *args, int, int);
cl_int ocl_shakedown(cpx *dev_in, cpx *dev_out, ocl_args *a_host, ocl_args *a_dev);
cl_int ocl_shakedown(cpx *dev_in, cpx *dev_out, ocl_args *a_host, ocl_args *a_dev, ocl_args *a_trans);
int ocl_free(cpx **din, cpx **dout, cpx **dref, const int n);
void ocl_setup_buffers(cpx **in, cpx **out, cpx **ref, const int n);
double ocl_get_elapsed(cl_event s, cl_event e);
cl_int ocl_get_platform(cl_platform_id *platform_id);

#endif