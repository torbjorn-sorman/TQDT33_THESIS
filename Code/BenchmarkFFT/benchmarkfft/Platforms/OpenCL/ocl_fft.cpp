#include "ocl_fft.h"

__inline cl_int runCombine(oclArgs *argCPU, oclArgs *argGPU);
__inline cl_int runCombine2D(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans);

//
// 1D
//

bool OCL_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data_in = get_seq(n, 1);
    cpx *data_out = get_seq(n);
    cpx *data_ref = get_seq(n, data_in);

    oclArgs argGPU, argCPU;
    checkErr(oclCreateKernels(&argCPU, &argGPU, data_in, FFT_FORWARD, n), "Create failed!");
    checkErr(runCombine(&argCPU, &argGPU), "Run failed");
    checkErr(oclRelease(data_out, &argCPU, &argGPU), "Release failed!");

    for (int i = 0; i < n; ++i) {
        printf("%f\t%f\n", data_out[i].x, data_out[i].y);
    }
    getchar();

    checkErr(oclCreateKernels(&argCPU, &argGPU, data_out, FFT_INVERSE, n), "Create failed!");
    runCombine(&argCPU, &argGPU);
    checkErr(oclRelease(data_in, &argCPU, &argGPU), "Release failed!");

    return freeResults(&data_in, &data_out, &data_ref, n) == 0;
}

double OCL_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_PERFORMANCE];
    cpx *data_in = get_seq(n, 1);
    oclArgs argGPU, argCPU;
    checkErr(oclCreateKernels(&argCPU, &argGPU, data_in, FFT_FORWARD, n), "Create failed!");
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        runCombine(&argCPU, &argGPU);
        measurements[i] = stopTimer();
    }
    checkErr(oclRelease(data_in, &argCPU, &argGPU), "Release failed!");
    int res = freeResults(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_PERFORMANCE);
}

//
// 2D
//

bool OCL2D_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data, *data_ref;
    setupBuffers(&data, NULL, &data_ref, n);
    write_image("OpenCL", "original", data, n);
    {
        oclArgs argGPU, argCPU, argTranspose;
        checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data, FFT_FORWARD, n), "Create failed!");
        checkErr(runCombine2D(&argCPU, &argGPU, &argTranspose), "Run failed!");
        checkErr(oclRelease2D(NULL, data, &argCPU, &argGPU, &argTranspose), "Release failed!");
        write_normalized_image("OpenCL", "frequency", data, n, true);
        write_image("OpenCL", "frequency - not norm", data, n);
    }
    {
        oclArgs argGPU, argCPU, argTranspose;
        checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data, FFT_INVERSE, n), "Create failed!");
        checkErr(runCombine2D(&argCPU, &argGPU, &argTranspose), "Run failure!");
        checkErr(oclRelease2D(NULL, data, &argCPU, &argGPU, &argTranspose), "Release failed!");
        write_image("OpenCL", "spatial", data, n);
    }
    return freeResults(&data, NULL, &data_ref, n) == 0;
}

double OCL2D_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[NUM_PERFORMANCE];
    int minDim = n < TILE_DIM ? TILE_DIM * TILE_DIM : n * n;
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * minDim);

    oclArgs argGPU, argCPU, argTranspose;    
    checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data_in, FFT_FORWARD, n), "Create failed!");

    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        runCombine2D(&argCPU, &argGPU, &argTranspose);
        measurements[i] = stopTimer();
    }

    checkErr(oclRelease2D(data_in, NULL, &argCPU, &argGPU, &argTranspose), "Release failed!");

    int res = freeResults(&data_in, NULL, NULL, n);
    return average_best(measurements, NUM_PERFORMANCE);
}

//
// Algorithm
//

__inline cl_int runCombineHelper(oclArgs *argCPU, oclArgs *argGPU, cl_mem in, cl_mem out)
{
    int n = argGPU->n;
    int n_half = (n / 2);
    int steps_left = log2_32(n);
    int leading_bits = 32 - steps_left;
    int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    cpx scale = make_cuFloatComplex((argGPU->dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    int number_of_blocks = argGPU->global_work_size[0];
    int local_n = n / argGPU->global_work_size[0];
    float global_angle = argGPU->dir * (M_2_PI / n);
    float local_angle = argGPU->dir * (M_2_PI / local_n);
    int range = n_half;
    if (number_of_blocks > HW_LIMIT) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --steps_left;
        int steps = 0;
        int dist = n_half;
        oclSetKernelCPUArg(argCPU, in, out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
        oclExecute(argCPU);
        // Instead of swapping input/output, run in place. The argGPU kernel needs to swap once.                
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            oclSetKernelCPUArg(argCPU, out, out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
            oclExecute(argCPU);
        }
        ++steps_left;
        range = local_n / 2;
        number_of_blocks = 1;
        //swap(&in, &out);
        in = out;
    }
    // Calculate complete sequence in one launch and syncronize steps on GPU
    oclSetKernelGPUArg(argGPU, in, out, global_angle, local_angle, steps_left, leading_bits, steps_gpu, scale, local_n, range);
    return oclExecute(argGPU);
}
/*
if (blocks >= HW_LIMIT) {
swapBuffer(dev_in, dev_out);
++steps_left;
bSize = n_per_block / 2;
number_of_blocks = 1;
}

// Calculate complete sequence in one launch and syncronize on GPU
cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
cuda_kernel_local KERNEL_ARGS3(blocks, threads, sizeof(cpx) * n_per_block) (*dev_in, *dev_out, global_angle, dir * (M_2_PI / n_per_block), steps_left, leading_bits, steps_gpu, scaleCpx, number_of_blocks, bSize);
cudaDeviceSynchronize();
*/


__inline cl_int runCombineHelper(oclArgs *argCPU, oclArgs *argGPU, cl_mem in, cl_mem out, int number_of_blocks)
{
    int steps_left = log2_32(argCPU->n);
    const int leading_bits = 32 - steps_left;
    const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = { (argCPU->dir == FFT_FORWARD ? 1.f : 1.f / argCPU->n), 0.f };
    const int n_per_block = argCPU->n / number_of_blocks;
    const float global_angle = argCPU->dir * (M_2_PI / argCPU->n);
    const float w_bangle = argCPU->dir * (M_2_PI / n_per_block);
    int bSize = argCPU->n;

    if (number_of_blocks > 1) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --steps_left;
        int steps = 0;
        int dist = argGPU->n / 2;
        oclSetKernelCPUArg(argCPU, in, out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
        checkErr(oclExecute(argCPU), "CPU Sync Kernel");
        // Instead of swapping input/output, run in place. The argGPU kernel needs to swap once.                
        while (--steps_left > steps_gpu) {
            dist >>= 1;
            ++steps;
            oclSetKernelCPUArg(argCPU, out, out, global_angle, 0xFFFFFFFF << steps_left, steps, dist);
            checkErr(oclExecute(argCPU), "CPU Sync Kernel");
        }
        ++steps_left;
        bSize = n_per_block;        
        number_of_blocks = 1; 
        
            swap(&in, &out);
    }
    // Calculate complete sequence in one launch and syncronize steps on GPU    
    oclSetKernelGPU2DArg(argGPU, in, out, w_bangle, steps_left, scaleCpx, bSize);    
    return oclExecute(argGPU);
}

__inline cl_int runCombine(oclArgs *argCPU, oclArgs *argGPU)
{   
    return runCombineHelper(argCPU, argGPU, argGPU->input, argGPU->output);
}

__inline cl_int runCombine2D(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans)
{    
    cl_mem _in = argGPU->input;
    cl_mem _out = argGPU->output;    
    // _in -> _out
    checkErr(runCombineHelper(argCPU, argGPU, _in, _out, (int)argGPU->global_work_size[1]), "Helper 2D");    

    return CL_SUCCESS;

    // _out -> _in
    oclSetKernelTransposeArg(argTrans, _out, _in);    
    checkErr(oclExecute(argTrans), "Transpose");    
    // _in -> _out    
    checkErr(runCombineHelper(argCPU, argGPU, _in, _out, argGPU->global_work_size[1]), "Helper 2D 2");
    // _out -> _in
    oclSetKernelTransposeArg(argTrans, _out, _in);    
    checkErr(oclExecute(argTrans), "Transpose 2");

    argCPU->input = argGPU->input = _out;
    argCPU->output = argGPU->output = _in;
    return CL_SUCCESS;
}
