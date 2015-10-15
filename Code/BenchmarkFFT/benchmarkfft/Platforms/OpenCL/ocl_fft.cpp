#include "ocl_fft.h"

__inline cl_int opencl_fft(oclArgs *argCPU, oclArgs *argGPU);
__inline cl_int opencl_fft_2d(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argGPUCol, oclArgs *argTrans);

//
// 1D
//

bool opencl_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data_in = get_seq(n, 1);
    cpx *data_out = get_seq(n);
    cpx *data_ref = get_seq(n, data_in);
    {
        oclArgs argGPU, argCPU;
        checkErr(opencl_create_kernels(&argCPU, &argGPU, data_in, FFT_FORWARD, n), "Create failed!");
        checkErr(opencl_fft(&argCPU, &argGPU), "Run failed");
        checkErr(oclRelease(data_out, &argCPU, &argGPU), "Release failed!");
    }
    double diff = diff_forward_sinus(data_out, n);
    if ((diff / (n >> 1)) > RELATIVE_ERROR_MARGIN) {
        printf("(%f)", diff);
        return false && freeResults(&data_in, &data_out, &data_ref, n);
    }
    {
        oclArgs argGPU, argCPU;
        checkErr(opencl_create_kernels(&argCPU, &argGPU, data_out, FFT_INVERSE, n), "Create failed!");
        checkErr(opencl_fft(&argCPU, &argGPU), "Run Inverse Failed");
        checkErr(oclRelease(data_in, &argCPU, &argGPU), "Release failed!");
    }

    return freeResults(&data_in, &data_out, &data_ref, n) == 0;
}

double opencl_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_PERFORMANCE];
    cpx *data_in = get_seq(n, 1);
    oclArgs argGPU, argCPU;
    checkErr(opencl_create_kernels(&argCPU, &argGPU, data_in, FFT_FORWARD, n), "Create failed!");
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        opencl_fft(&argCPU, &argGPU);
        measurements[i] = stopTimer();
    }
    checkErr(oclRelease(data_in, &argCPU, &argGPU), "Release failed!");
    int res = freeResults(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_PERFORMANCE);
}

//
// 2D
//

bool opencl_2d_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data, *data_ref;
    setupBuffers(&data, NULL, &data_ref, n);
    write_image("OpenCL", "original", data, n);
    {
        oclArgs argGPU, argCPU, argGPUCol, argTranspose;
        checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argGPUCol, &argTranspose, data, FFT_FORWARD, n), "Create failed!");
        checkErr(opencl_fft_2d(&argCPU, &argGPU, &argGPUCol, &argTranspose), "Run failed!");
        checkErr(oclRelease2D(NULL, data, &argCPU, &argGPU, &argGPUCol, &argTranspose), "Release failed!");
        write_normalized_image("OpenCL", "frequency", data, n, true);
        write_image("OpenCL", "frequency - not norm", data, n);
    }
    {
        oclArgs argGPU, argCPU, argGPUCol, argTranspose;
        checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argGPUCol, &argTranspose, data, FFT_INVERSE, n), "Create failed!");
        checkErr(opencl_fft_2d(&argCPU, &argGPU, &argGPUCol, &argTranspose), "Run failure!");
        checkErr(oclRelease2D(NULL, data, &argCPU, &argGPU, &argGPUCol, &argTranspose), "Release failed!");
        write_image("OpenCL", "spatial", data, n);
    }
    return freeResults(&data, NULL, &data_ref, n) == 0;
}

double opencl_2d_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_PERFORMANCE];
    int minDim = n < TILE_DIM ? TILE_DIM * TILE_DIM : n * n;
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * minDim);

    oclArgs argGPU, argCPU, argGPUCol, argTranspose;
    checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argGPUCol, &argTranspose, data_in, FFT_FORWARD, n), "Create failed!");

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        opencl_fft_2d(&argCPU, &argGPU, &argGPUCol, &argTranspose);
        measurements[i] = stopTimer();
    }

    checkErr(oclRelease2D(data_in, NULL, &argCPU, &argGPU, &argGPUCol, &argTranspose), "Release failed!");

    int res = freeResults(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_PERFORMANCE);
}

//
// Algorithm
//

__inline cl_int opencl_fft(oclArgs *argCPU, oclArgs *argGPU)
{   
    cl_mem in = argGPU->input;
    cl_mem out = argGPU->output;
    int n = argGPU->n;
    int n_half = (n >> 1);
    int steps_left = log2_32(n);
    int leading_bits = 32 - steps_left;
    int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    float scalar = (argGPU->dir == FFT_FORWARD ? 1.f : 1.f / n);
    
    int number_of_blocks = (int)(argGPU->global_work_size[0] / argGPU->local_work_size[0]);
    int n_per_block = n / number_of_blocks;
    float global_angle = argGPU->dir * (M_2_PI / n);
    float local_angle = argGPU->dir * (M_2_PI / n_per_block);
    int block_range_half = n_half;
    if (number_of_blocks >= HW_LIMIT) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.     

        --steps_left;
        int steps = 0;
        int dist = n_half;
        opencl_set_kernel_args_global(argCPU, &in, &out, &global_angle, 0xFFFFFFFF << steps_left, &steps, &dist);
        checkErr(opencl_execute(argCPU), "Failed Global in -> out");
        // Instead of swapping input/output, run in place. The argGPU kernel needs to swap once.                
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            opencl_set_kernel_args_global(argCPU, &out, &out, &global_angle, 0xFFFFFFFF << steps_left, &steps, &dist);
            checkErr(opencl_execute(argCPU), "Failed Global out -> out");
        }
        in = out;
        ++steps_left;
        number_of_blocks = 1;
        block_range_half = n_per_block >> 1;
    }
                                                  
    opencl_set_kernel_args_local(argGPU, in, out, global_angle, local_angle, steps_left, leading_bits, steps_gpu, scalar, number_of_blocks, block_range_half);
    return opencl_execute(argGPU);
}

__inline cl_int runCombineHelper(oclArgs *argCPU, oclArgs *argGPU, cl_mem in, cl_mem out, int number_of_blocks)
{


    int steps_left = log2_32(argCPU->n);
    const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    float scalar = (argCPU->dir == FFT_FORWARD ? 1.f : 1.f / argCPU->n);

    const int n_per_block = argCPU->n / number_of_blocks;
    const float global_angle = argCPU->dir * (M_2_PI / argCPU->n);
    const float local_angle = argCPU->dir * (M_2_PI / n_per_block);
    int block_range_half = argCPU->n;
    if (number_of_blocks > 1) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --steps_left;
        int steps = 0;
        int dist = argGPU->n >> 1;
        opencl_set_kernel_args_global(argCPU, &in, &out, &global_angle, 0xFFFFFFFF << steps_left, &steps, &dist);
        checkErr(opencl_execute(argCPU), "CPU Sync Kernel");
        // Instead of swapping input/output, run in place. The argGPU kernel needs to swap once.                
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            opencl_set_kernel_args_global(argCPU, &out, &out, &global_angle, 0xFFFFFFFF << steps_left, &steps, &dist);
            checkErr(opencl_execute(argCPU), "CPU Sync Kernel");
        }
        in = out;        
        ++steps_left;
        block_range_half = n_per_block;
    }
    // Calculate complete sequence in one launch and syncronize steps on GPU    
    oclSetKernelGPU2DArg(argGPU, in, out, local_angle, steps_left, scalar, block_range_half);
    return opencl_execute(argGPU);
}

__inline cl_int opencl_fft_2d(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argGPUCol, oclArgs *argTrans)
{    
    const cl_mem _in = argGPU->input;
    const cl_mem _out = argGPU->output;    
    if (argGPU->n > 256) {
        // _in -> _out
        checkErr(runCombineHelper(argCPU, argGPU, _in, _out, (int)argGPU->global_work_size[1]), "Helper 2D");        
        // _out -> _in
        oclSetKernelTransposeArg(argTrans, _out, _in);    
        checkErr(opencl_execute(argTrans), "Transpose");    
        // _in -> _out    
        checkErr(runCombineHelper(argCPU, argGPU, _in, _out, (int)argGPU->global_work_size[1]), "Helper 2D 2");
        // _out -> _in
        oclSetKernelTransposeArg(argTrans, _out, _in);    
        checkErr(opencl_execute(argTrans), "Transpose 2");
    }
    else {
        const int steps_left = log2_32(argGPU->n);
        const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
        const float scalar = (argGPU->dir == FFT_FORWARD ? 1.f : 1.f / argGPU->n);
        const float global_angle = argGPU->dir * (M_2_PI / argGPU->n);
        // Calculate complete sequence in one launch and syncronize steps on GPU    
        oclSetKernelGPU2DArg(argGPU, _in, _out, global_angle, steps_left, scalar, argGPU->n);
        checkErr(opencl_execute(argGPU), "Rows");
        // Calculate complete sequence in one launch and syncronize steps on GPU    
        oclSetKernelGPU2DColArg(argGPUCol, _out, _in, global_angle, steps_left, scalar, argGPUCol->n);
        checkErr(opencl_execute(argGPUCol), "Cols");
    }
    argCPU->input = argGPUCol->input = argGPU->input = _out;
    argCPU->output = argGPUCol->output = argGPU->output = _in;
    return CL_SUCCESS;
}
