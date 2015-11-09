#include "ocl_fft.h"

__inline void ocl_fft(ocl_args *a_host, ocl_args *a_dev);
__inline void ocl_fft_2d(ocl_args *a_host, ocl_args *a_dev, ocl_args *a_trans);

#define OCL_GROUP_SIZE 512
#define OCL_TILE_DIM 32 // This is 8K local/shared mem
#define OCL_BLOCK_DIM 16 // This is 256 Work Items / Group

//
// 1D
//

bool ocl_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data = get_seq(n, 1);
    cpx *data_ref = get_seq(n, data);
    ocl_args a_dev, a_host;
    ocl_check_err(ocl_setup(&a_host, &a_dev, data, FFT_FORWARD, OCL_GROUP_SIZE, n), "Create failed!");

    ocl_fft(&a_host, &a_dev);
    clFinish(a_dev.commands);
    ocl_check_err(clEnqueueReadBuffer(a_dev.commands, a_dev.output, CL_TRUE, 0, a_dev.data_mem_size, data, 0, NULL, NULL), "Read output for validation failed!");
    double diff = diff_forward_sinus(data, n);

    a_dev.dir = a_host.dir = FFT_INVERSE;
    swap(&a_dev.input, &a_dev.output);
    ocl_fft(&a_host, &a_dev);
    clFinish(a_dev.commands);
    ocl_check_err(ocl_shakedown(NULL, data, &a_host, &a_dev), "Release failed!");
    return (ocl_free(&data, NULL, &data_ref, n) == 0) && (diff <= RELATIVE_ERROR_MARGIN);
}

bool ocl_2d_validate(const int n, bool write_img)
{
    cl_int err = CL_SUCCESS;
    cpx *data, *data_ref;
    ocl_setup_buffers(&data, NULL, &data_ref, n);
    ocl_args a_dev, a_host, argTranspose;
    ocl_check_err(ocl_setup(&a_host, &a_dev, &argTranspose, data, FFT_FORWARD, OCL_GROUP_SIZE, OCL_TILE_DIM, OCL_BLOCK_DIM, n), "Create failed!");

    ocl_fft_2d(&a_host, &a_dev, &argTranspose);
    clFinish(a_dev.commands);

    if (write_img) {
        ocl_check_err(clEnqueueReadBuffer(a_dev.commands, a_dev.output, CL_TRUE, 0, a_dev.data_mem_size, data, 0, NULL, NULL), "Read Output Buffer!");
        write_normalized_image("OpenCL", "freq", data, n, true);
    }

    a_dev.dir = a_host.dir = FFT_INVERSE;
    swap(&a_dev.input, &a_dev.output);
    ocl_fft_2d(&a_host, &a_dev, &argTranspose);
    clFinish(a_dev.commands);
    ocl_check_err(ocl_shakedown(NULL, data, &a_host, &a_dev, &argTranspose), "Release failed!");

    if (write_img) {
        write_image("OpenCL", "spat", data, n);
    }
    return ocl_free(&data, NULL, &data_ref, n) == 0;
}

#ifndef MEASURE_BY_TIMESTAMP
double ocl_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    cpx *data_in = get_seq(n, 1);
    ocl_args a_dev, a_host;
    ocl_check_err(ocl_setup(&a_host, &a_dev, data_in, FFT_FORWARD, OCL_GROUP_SIZE, n), "Create failed!");

    for (int i = 0; i < NUM_TESTS; ++i) {
        start_timer();
        ocl_fft(&a_host, &a_dev);
        clFinish(a_dev.commands);
        measurements[i] = stop_timer();
    }
    ocl_check_err(ocl_shakedown(data_in, NULL, &a_host, &a_dev), "Release failed!");
    int res = ocl_free(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_TESTS);
}
double ocl_2d_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * n * n);
    ocl_args a_dev, a_host, argTranspose;
    ocl_check_err(ocl_setup(&a_host, &a_dev, &argTranspose, data_in, FFT_FORWARD, OCL_GROUP_SIZE, OCL_TILE_DIM, OCL_BLOCK_DIM, n), "Create failed!");
    for (int i = 0; i < NUM_TESTS; ++i) {
        start_timer();
        ocl_fft_2d(&a_host, &a_dev, &argTranspose);
        clFinish(a_dev.commands);
        measurements[i] = stop_timer();
    }
    ocl_check_err(ocl_shakedown(data_in, NULL, &a_host, &a_dev, &argTranspose), "Release failed!");
    int res = ocl_free(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_TESTS);
}
#else
double ocl_get_elapsed(cl_event s, cl_event e)
{
    cl_ulong start = 0, end = 0;
    clWaitForEvents(1, &e);
    clGetEventProfilingInfo(s, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    return (double)(end - start)*(cl_double)(1e-03);
}
double ocl_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    ocl_args a_dev, a_host, arg_timestamp;
    ocl_check_err(ocl_setup(&a_host, &a_dev, NULL, FFT_FORWARD, OCL_GROUP_SIZE, n), "Create failed!");
    ocl_setup_timestamp(&a_dev, &arg_timestamp);
    cl_event start_event, end_event;
    clFinish(a_dev.commands);
    for (int i = 0; i < NUM_TESTS; ++i) {        
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.work_size, arg_timestamp.group_work_size, 0, NULL, &start_event);
        ocl_fft(&a_host, &a_dev);
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.work_size, arg_timestamp.group_work_size, 0, NULL, &end_event);
        measurements[i] = ocl_get_elapsed(start_event, end_event);        
    }
    ocl_check_err(ocl_shakedown(NULL, NULL, &a_host, &a_dev), "Release failed!");
    return average_best(measurements, NUM_TESTS);
}
double ocl_2d_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    int minDim = n < OCL_TILE_DIM ? OCL_TILE_DIM * OCL_TILE_DIM : n * n;
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * minDim);
    ocl_args a_dev, a_host, argTranspose, arg_timestamp;
    ocl_check_err(ocl_setup(&a_host, &a_dev, &argTranspose, data_in, FFT_FORWARD, OCL_GROUP_SIZE, OCL_TILE_DIM, OCL_BLOCK_DIM, n), "Create failed!");
    ocl_setup_timestamp(&a_dev, &arg_timestamp);
    cl_event start_event, end_event;
    clFinish(a_dev.commands);
    for (int i = 0; i < NUM_TESTS; ++i) {
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.work_size, arg_timestamp.group_work_size, 0, NULL, &start_event);
        ocl_fft_2d(&a_host, &a_dev, &argTranspose);
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.work_size, arg_timestamp.group_work_size, 0, NULL, &end_event);        
        measurements[i] = ocl_get_elapsed(start_event, end_event);
    }
    ocl_check_err(ocl_shakedown(data_in, NULL, &a_host, &a_dev, &argTranspose), "Release failed!");
    int res = ocl_free(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_TESTS);
}
#endif

// ---------------------------------
//
// Algorithm
//
// ---------------------------------

__inline void ocl_fft(ocl_args *a_host, ocl_args *a_dev)
{
    fft_args args;
    set_fft_arguments(&args, a_dev->dir, a_dev->number_of_blocks, OCL_GROUP_SIZE, a_dev->n);
    if (a_dev->number_of_blocks > 1) {
        while (--args.steps_left > args.steps_gpu) {
            ocl_set_args(a_host, a_dev->input, args.global_angle, 0xFFFFFFFF << args.steps_left, args.steps++, args.dist >>= 1);
            clEnqueueNDRangeKernel(a_host->commands, a_host->kernel, a_host->workDim, NULL, a_host->work_size, a_host->group_work_size, 0, NULL, NULL);
        }
        ++args.steps_left;
    }
    ocl_set_args(a_dev, a_dev->input, a_dev->output, args.local_angle, args.steps_left, args.leading_bits, args.scalar, args.block_range);
    clEnqueueNDRangeKernel(a_dev->commands, a_dev->kernel, a_dev->workDim, NULL, a_dev->work_size, a_dev->group_work_size, 0, NULL, NULL);
}

__inline void ocl_fft_2d(ocl_args *a_host, ocl_args *a_dev, ocl_args *a_trans)
{
    cl_mem _in = a_dev->input;
    cl_mem _out = a_dev->output;

    ocl_fft(a_host, a_dev);    
    ocl_set_args(a_trans, _out, _in);
    clEnqueueNDRangeKernel(a_trans->commands, a_trans->kernel, a_trans->workDim, NULL, a_trans->work_size, a_trans->group_work_size, 0, NULL, NULL);
    
    ocl_fft(a_host, a_dev);    
    ocl_set_args(a_trans, _out, _in);
    clEnqueueNDRangeKernel(a_trans->commands, a_trans->kernel, a_trans->workDim, NULL, a_trans->work_size, a_trans->group_work_size, 0, NULL, NULL);

    a_host->input = a_dev->input = _out;
    a_host->output = a_dev->output = _in;
}