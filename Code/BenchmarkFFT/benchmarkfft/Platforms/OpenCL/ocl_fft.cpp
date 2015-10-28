#include "ocl_fft.h"

__inline void opencl_fft(oclArgs *arg_cpu, oclArgs *arg_gpu);
__inline void opencl_fft_2d(oclArgs *arg_cpu, oclArgs *arg_gpu, oclArgs *arg_transpose);

//
// 1D
//

bool opencl_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data = get_seq(n, 1);
    cpx *data_ref = get_seq(n, data);
    oclArgs arg_gpu, arg_cpu;
    checkErr(opencl_create_kernels(&arg_cpu, &arg_gpu, data, FFT_FORWARD, n), "Create failed!");
    
    opencl_fft(&arg_cpu, &arg_gpu);
    clFinish(arg_gpu.commands);
    checkErr(clEnqueueReadBuffer(arg_gpu.commands, arg_gpu.output, CL_TRUE, 0, arg_gpu.data_mem_size, data, 0, NULL, NULL), "Read output for validation failed!");
    double diff = diff_forward_sinus(data, n);
    
    arg_gpu.dir = arg_cpu.dir = FFT_INVERSE;    
    swap(&arg_gpu.input, &arg_gpu.output);
    opencl_fft(&arg_cpu, &arg_gpu);
    clFinish(arg_gpu.commands);
    checkErr(oclRelease(NULL, data, &arg_cpu, &arg_gpu), "Release failed!");
    return (freeResults(&data, NULL, &data_ref, n) == 0) && (diff <= RELATIVE_ERROR_MARGIN);
}

bool opencl_2d_validate(const int n, bool write_img)
{
    cl_int err = CL_SUCCESS;
    cpx *data, *data_ref;
    setupBuffers(&data, NULL, &data_ref, n);
    oclArgs arg_gpu, arg_cpu, argTranspose;
    checkErr(oclCreateKernels2D(&arg_cpu, &arg_gpu, &argTranspose, data, FFT_FORWARD, n), "Create failed!");

    opencl_fft_2d(&arg_cpu, &arg_gpu, &argTranspose);
    clFinish(arg_gpu.commands);    
    
    if (write_img) {
        checkErr(clEnqueueReadBuffer(arg_gpu.commands, arg_gpu.input, CL_TRUE, 0, arg_gpu.data_mem_size, data, 0, NULL, NULL), "Read Output Buffer!");
        write_normalized_image("OpenCL", "freq", data, n, true);
    }
    
    arg_gpu.dir = arg_cpu.dir = FFT_INVERSE;
    swap(&arg_gpu.input, &arg_gpu.output);
    opencl_fft_2d(&arg_cpu, &arg_gpu, &argTranspose);
    clFinish(arg_gpu.commands);
    checkErr(oclRelease2D(NULL, data, &arg_cpu, &arg_gpu, &argTranspose), "Release failed!");
    
    if (write_img) {
        write_image("OpenCL", "spat", data, n);    
    }
    return freeResults(&data, NULL, &data_ref, n) == 0;
}

#ifndef MEASURE_BY_TIMESTAMP
double opencl_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    cpx *data_in = get_seq(n, 1);
    oclArgs arg_gpu, arg_cpu;
    checkErr(opencl_create_kernels(&arg_cpu, &arg_gpu, data_in, FFT_FORWARD, n), "Create failed!");
    for (int i = 0; i < NUM_TESTS; ++i) {
        startTimer();
        opencl_fft(&arg_cpu, &arg_gpu);

        clFinish(arg_gpu.commands);
        measurements[i] = stopTimer();
    }
    checkErr(oclRelease(data_in, NULL, &arg_cpu, &arg_gpu), "Release failed!");
    int res = freeResults(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_TESTS);
}
double opencl_2d_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    int minDim = n < TILE_DIM ? TILE_DIM * TILE_DIM : n * n;
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * minDim);
    oclArgs arg_gpu, arg_cpu, argTranspose;
    checkErr(oclCreateKernels2D(&arg_cpu, &arg_gpu, &argTranspose, data_in, FFT_FORWARD, n), "Create failed!");
    for (int i = 0; i < NUM_TESTS; ++i) {
        startTimer();
        opencl_fft_2d(&arg_cpu, &arg_gpu, &argTranspose);
        clFinish(arg_gpu.commands);
        measurements[i] = stopTimer();
    }
    checkErr(oclRelease2D(data_in, NULL, &arg_cpu, &arg_gpu, &argTranspose), "Release failed!");
    int res = freeResults(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_TESTS);
    }
#else
double opencl_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    oclArgs arg_gpu, arg_cpu, arg_timestamp;
    checkErr(opencl_create_kernels(&arg_cpu, &arg_gpu, NULL, FFT_FORWARD, n), "Create failed!");
    opencl_create_timestamp_kernel(&arg_gpu, &arg_timestamp);
    cl_event start_event, end_event;
    cl_ulong start = 0, end = 0;
    for (int i = 0; i < NUM_TESTS; ++i) {
        clFinish(arg_gpu.commands);
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.global_work_size, arg_timestamp.local_work_size, 0, NULL, &start_event);
        opencl_fft(&arg_cpu, &arg_gpu);
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.global_work_size, arg_timestamp.local_work_size, 0, NULL, &end_event);
        clWaitForEvents(1, &end_event);
        clGetEventProfilingInfo(start_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(end_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        measurements[i] = (double)(end - start)*(cl_double)(1e-03);
        clFinish(arg_gpu.commands);
    }
    checkErr(oclRelease(NULL, NULL, &arg_cpu, &arg_gpu), "Release failed!");
    return average_best(measurements, NUM_TESTS);
}
double opencl_2d_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_TESTS];
    int minDim = n < TILE_DIM ? TILE_DIM * TILE_DIM : n * n;
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * minDim);
    oclArgs arg_gpu, arg_cpu, argTranspose, arg_timestamp;
    checkErr(oclCreateKernels2D(&arg_cpu, &arg_gpu, &argTranspose, data_in, FFT_FORWARD, n), "Create failed!");
    opencl_create_timestamp_kernel(&arg_gpu, &arg_timestamp);
    cl_event start_event, end_event;
    cl_ulong start = 0, end = 0;
    clFinish(arg_gpu.commands);
    for (int i = 0; i < NUM_TESTS; ++i) {        
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.global_work_size, arg_timestamp.local_work_size, 0, NULL, &start_event);
        opencl_fft_2d(&arg_cpu, &arg_gpu, &argTranspose);
        clEnqueueNDRangeKernel(arg_timestamp.commands, arg_timestamp.kernel, arg_timestamp.workDim, NULL, arg_timestamp.global_work_size, arg_timestamp.local_work_size, 0, NULL, &end_event);
        clWaitForEvents(1, &end_event);
        clFinish(arg_gpu.commands);
        clGetEventProfilingInfo(start_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(end_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        measurements[i] = (double)(end - start)*(cl_double)(1e-03);
    }
    checkErr(oclRelease2D(data_in, NULL, &arg_cpu, &arg_gpu, &argTranspose), "Release failed!");
    int res = freeResults(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_TESTS);
}
#endif

// ---------------------------------
//
// Algorithm
//
// ---------------------------------

__inline void opencl_fft(oclArgs *arg_cpu, oclArgs *arg_gpu) //, cl_event *events)
{
    const int n = arg_gpu->n;
    int steps_left = log2_32(n);
    const int leading_bits = 32 - steps_left;
    const float scalar = (arg_gpu->dir == FFT_FORWARD ? 1.f : 1.f / n);
    int number_of_blocks = (int)(arg_gpu->global_work_size[0] / arg_gpu->local_work_size[0]);
    const int n_per_block = n / number_of_blocks;
    const float local_angle = arg_gpu->dir * (M_2_PI / n_per_block);
    int block_range_half = (n >> 1);
    if (number_of_blocks > 1) {
        const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
        const float global_angle = arg_gpu->dir * (M_2_PI / n);
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.        
        int steps = 0;
        int dist = n;
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            opencl_set_kernel_args_global(arg_cpu, arg_gpu->input, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
            clEnqueueNDRangeKernel(arg_cpu->commands, arg_cpu->kernel, arg_cpu->workDim, NULL, arg_cpu->global_work_size, arg_cpu->local_work_size, 0, NULL, NULL);
            ++steps;
        }
        ++steps_left;
        block_range_half = n_per_block >> 1;
    }
    // Calculate complete sequence in one launch and syncronize steps on GPU  
    opencl_set_kernel_args_local(arg_gpu, arg_gpu->input, arg_gpu->output, local_angle, steps_left, leading_bits, scalar, block_range_half);
    clEnqueueNDRangeKernel(arg_gpu->commands, arg_gpu->kernel, arg_gpu->workDim, NULL, arg_gpu->global_work_size, arg_gpu->local_work_size, 0, NULL, NULL);
}

__inline void opencl_fft_2d_helper(oclArgs *arg_cpu, oclArgs *arg_gpu, cl_mem *in, cl_mem *out, int number_of_blocks)
{
    int steps_left = log2_32(arg_cpu->n);
    const int leading_bits = 32 - steps_left;
    const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    float scalar = (arg_cpu->dir == FFT_FORWARD ? 1.f : 1.f / arg_cpu->n);
    const int n_per_block = arg_cpu->n / number_of_blocks;
    const float global_angle = arg_cpu->dir * (M_2_PI / arg_cpu->n);
    const float local_angle = arg_cpu->dir * (M_2_PI / n_per_block);
    int block_range = arg_cpu->n;
    if (number_of_blocks > 1) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.        
        int steps = 0;
        int dist = arg_gpu->n;
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            opencl_set_kernel_args_global(arg_cpu, *in, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
            clEnqueueNDRangeKernel(arg_cpu->commands, arg_cpu->kernel, arg_cpu->workDim, NULL, arg_cpu->global_work_size, arg_cpu->local_work_size, 0, NULL, NULL);
            ++steps;
        }
        ++steps_left;
        block_range = n_per_block;
    }
    // Calculate complete sequence in one launch and syncronize steps on GPU    
    oclSetKernelGPU2DArg(arg_gpu, *in, *out, local_angle, steps_left, leading_bits, scalar, block_range);
    clEnqueueNDRangeKernel(arg_gpu->commands, arg_gpu->kernel, arg_gpu->workDim, NULL, arg_gpu->global_work_size, arg_gpu->local_work_size, 0, NULL, NULL);
}

__inline void opencl_fft_2d(oclArgs *arg_cpu, oclArgs *arg_gpu_row, oclArgs *arg_transpose)
{
    cl_mem _in = arg_gpu_row->input;
    cl_mem _out = arg_gpu_row->output;
        
    int number_of_blocks = (int)arg_gpu_row->global_work_size[1];
    // _in -> _out
    opencl_fft_2d_helper(arg_cpu, arg_gpu_row, &_in, &_out, number_of_blocks);
    // _out -> _in
    oclSetKernelTransposeArg(arg_transpose, _out, _in);
    clEnqueueNDRangeKernel(arg_transpose->commands, arg_transpose->kernel, arg_transpose->workDim, NULL, arg_transpose->global_work_size, arg_transpose->local_work_size, 0, NULL, NULL);
    // _in -> _out    
    opencl_fft_2d_helper(arg_cpu, arg_gpu_row, &_in, &_out, number_of_blocks);
    // _out -> _in
    oclSetKernelTransposeArg(arg_transpose, _out, _in);
    clEnqueueNDRangeKernel(arg_transpose->commands, arg_transpose->kernel, arg_transpose->workDim, NULL, arg_transpose->global_work_size, arg_transpose->local_work_size, 0, NULL, NULL);

    arg_cpu->input = arg_gpu_row->input = _out;
    arg_cpu->output = arg_gpu_row->output = _in;
}