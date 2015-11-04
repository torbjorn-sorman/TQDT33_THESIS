#include "ocl_fft.h"

__inline void ocl_fft(ocl_args *a_cpu, ocl_args *a_gpu);
__inline void ocl_fft_2d(ocl_args *a_cpu, ocl_args *a_gpu, ocl_args *a_trans);

//
// 1D
//

bool ocl_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data = get_seq(n, 1);
    cpx *data_ref = get_seq(n, data);
    ocl_args a_gpu, a_cpu;
    ocl_check_err(ocl_setup(&a_cpu, &a_gpu, data, FFT_FORWARD, n), "Create failed!");

    ocl_fft(&a_cpu, &a_gpu);
    clFinish(a_gpu.commands);
    ocl_check_err(clEnqueueReadBuffer(a_gpu.commands, a_gpu.output, CL_TRUE, 0, a_gpu.data_mem_size, data, 0, NULL, NULL), "Read output for validation failed!");
    double diff = diff_forward_sinus(data, n);

    a_gpu.dir = a_cpu.dir = FFT_INVERSE;
    swap(&a_gpu.input, &a_gpu.output);
    ocl_fft(&a_cpu, &a_gpu);
    clFinish(a_gpu.commands);
    ocl_check_err(ocl_shakedown(NULL, data, &a_cpu, &a_gpu), "Release failed!");
    return (ocl_free(&data, NULL, &data_ref, n) == 0) && (diff <= RELATIVE_ERROR_MARGIN);
}

bool ocl_2d_validate(const int n, bool write_img)
{
    cl_int err = CL_SUCCESS;
    cpx *data, *data_ref;
    ocl_setup_buffers(&data, NULL, &data_ref, n);
    ocl_args a_gpu, a_cpu, argTranspose;
    ocl_check_err(ocl_setup(&a_cpu, &a_gpu, &argTranspose, data, FFT_FORWARD, n), "Create failed!");

    ocl_fft_2d(&a_cpu, &a_gpu, &argTranspose);
    clFinish(a_gpu.commands);

    if (write_img) {
        ocl_check_err(clEnqueueReadBuffer(a_gpu.commands, a_gpu.output, CL_TRUE, 0, a_gpu.data_mem_size, data, 0, NULL, NULL), "Read Output Buffer!");
#ifndef TRANSPOSE_ONLY
        write_normalized_image("OpenCL", "freq", data, n, true);
#else
        write_image("OpenCL", "trans", data, n);
#endif
    }

    a_gpu.dir = a_cpu.dir = FFT_INVERSE;
    swap(&a_gpu.input, &a_gpu.output);
    ocl_fft_2d(&a_cpu, &a_gpu, &argTranspose);
    clFinish(a_gpu.commands);
    ocl_check_err(ocl_shakedown(NULL, data, &a_cpu, &a_gpu, &argTranspose), "Release failed!");

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
    ocl_args a_gpu, a_cpu;
    ocl_check_err(ocl_setup(&a_cpu, &a_gpu, data_in, FFT_FORWARD, n), "Create failed!");
    for (int i = 0; i < NUM_TESTS; ++i) {
        startTimer();
        ocl_fft(&a_cpu, &a_gpu);

        clFinish(a_gpu.commands);
        measurements[i] = stopTimer();
    }
    ocl_check_err(ocl_shakedown(data_in, NULL, &a_cpu, &a_gpu), "Release failed!");
    int res = ocl_free(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_TESTS);
}
double ocl_2d_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    int minDim = n < TILE_DIM ? TILE_DIM * TILE_DIM : n * n;
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * minDim);
    ocl_args a_gpu, a_cpu, argTranspose;
    ocl_check_err(ocl_setup(&a_cpu, &a_gpu, &argTranspose, data_in, FFT_FORWARD, n), "Create failed!");
    for (int i = 0; i < NUM_TESTS; ++i) {
        startTimer();
        ocl_fft_2d(&a_cpu, &a_gpu, &argTranspose);
        clFinish(a_gpu.commands);
        measurements[i] = stopTimer();
    }
    ocl_check_err(ocl_shakedown(data_in, NULL, &a_cpu, &a_gpu, &argTranspose), "Release failed!");
    int res = ocl_free(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_TESTS);
}
#else
double ocl_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    ocl_args a_gpu, a_cpu, arg_timestamp;
    ocl_check_err(ocl_setup(&a_cpu, &a_gpu, NULL, FFT_FORWARD, n), "Create failed!");
    ocl_setup_timestamp(&a_gpu, &arg_timestamp);
    cl_event start_event, end_event;
    cl_ulong start = 0, end = 0;
    for (int i = 0; i < NUM_TESTS; ++i) {
        clFinish(a_gpu.commands);
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.global_work_size, arg_timestamp.local_work_size, 0, NULL, &start_event);
        ocl_fft(&a_cpu, &a_gpu);
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.global_work_size, arg_timestamp.local_work_size, 0, NULL, &end_event);
        clWaitForEvents(1, &end_event);
        clGetEventProfilingInfo(start_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(end_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        measurements[i] = (double)(end - start)*(cl_double)(1e-03);
        clFinish(a_gpu.commands);
    }
    ocl_check_err(ocl_shakedown(NULL, NULL, &a_cpu, &a_gpu), "Release failed!");
    return average_best(measurements, NUM_TESTS);
}
double ocl_2d_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    int minDim = n < OCL_TILE_DIM ? OCL_TILE_DIM * OCL_TILE_DIM : n * n;
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * minDim);
    ocl_args a_gpu, a_cpu, argTranspose, arg_timestamp;
    ocl_check_err(ocl_setup(&a_cpu, &a_gpu, &argTranspose, data_in, FFT_FORWARD, n), "Create failed!");
    ocl_setup_timestamp(&a_gpu, &arg_timestamp);
    cl_event start_event, end_event;
    cl_ulong start = 0, end = 0;
    clFinish(a_gpu.commands);
    for (int i = 0; i < NUM_TESTS; ++i) {
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.global_work_size, arg_timestamp.local_work_size, 0, NULL, &start_event);
        ocl_fft_2d(&a_cpu, &a_gpu, &argTranspose);
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.global_work_size, arg_timestamp.local_work_size, 0, NULL, &end_event);
        clWaitForEvents(1, &end_event);
        clFinish(a_gpu.commands);
        clGetEventProfilingInfo(start_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(end_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        measurements[i] = (double)(end - start)*(cl_double)(1e-03);
    }
    ocl_check_err(ocl_shakedown(data_in, NULL, &a_cpu, &a_gpu, &argTranspose), "Release failed!");
    int res = ocl_free(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_TESTS);
}
#endif

// ---------------------------------
//
// Algorithm
//
// ---------------------------------

__inline void ocl_fft(ocl_args *a_cpu, ocl_args *a_gpu)
{
    fft_args args;
    set_fft_arguments(&args, a_gpu->dir, a_gpu->number_of_blocks, OCL_GROUP_SIZE, a_gpu->n);
    if (a_gpu->number_of_blocks > 1) {
        while (--args.steps_left > args.steps_gpu) {
            ocl_set_args(a_cpu, a_gpu->input, args.global_angle, 0xFFFFFFFF << args.steps_left, args.steps++, args.dist >>= 1);
            clEnqueueNDRangeKernel(a_cpu->commands, a_cpu->kernel, a_cpu->workDim, NULL, a_cpu->global_work_size, a_cpu->local_work_size, 0, NULL, NULL);
        }
        ++args.steps_left;
    }
    ocl_set_args(a_gpu, a_gpu->input, a_gpu->output, args.local_angle, args.steps_left, args.leading_bits, args.scalar, args.block_range_half);
    clEnqueueNDRangeKernel(a_gpu->commands, a_gpu->kernel, a_gpu->workDim, NULL, a_gpu->global_work_size, a_gpu->local_work_size, 0, NULL, NULL);
}

__inline void ocl_fft_2d(ocl_args *a_cpu, ocl_args *a_gpu, ocl_args *a_trans)
{
    cl_mem _in = a_gpu->input;
    cl_mem _out = a_gpu->output;

    ocl_fft(a_cpu, a_gpu);    
    ocl_set_args(a_trans, _out, _in);
    clEnqueueNDRangeKernel(a_trans->commands, a_trans->kernel, a_trans->workDim, NULL, a_trans->global_work_size, a_trans->local_work_size, 0, NULL, NULL);
    
    ocl_fft(a_cpu, a_gpu);    
    ocl_set_args(a_trans, _out, _in);
    clEnqueueNDRangeKernel(a_trans->commands, a_trans->kernel, a_trans->workDim, NULL, a_trans->global_work_size, a_trans->local_work_size, 0, NULL, NULL);

    a_cpu->input = a_gpu->input = _out;
    a_cpu->output = a_gpu->output = _in;
}