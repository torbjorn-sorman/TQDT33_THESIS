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

    checkErr(oclCreateKernels(&argCPU, &argGPU, data_out, FFT_INVERSE, n), "Create failed!");
    runCombine(&argCPU, &argGPU);
    checkErr(oclRelease(data_in, &argCPU, &argGPU), "Release failed!");

    return freeResults(&data_in, &data_out, &data_ref, n) == 0;
}

double OCL_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[20];
    cpx *data_in = get_seq(n, 1);
    oclArgs argGPU, argCPU;
    checkErr(oclCreateKernels(&argCPU, &argGPU, data_in, FFT_FORWARD, n), "Create failed!");
    for (int i = 0; i < 20; ++i) {
        startTimer();
        runCombine(&argCPU, &argGPU);
        measurements[i] = stopTimer();
    }
    checkErr(oclRelease(data_in, &argCPU, &argGPU), "Release failed!");
    int res = freeResults(&data_in, NULL, NULL, n);
    return avg(measurements, 20);
}

//
// 2D
//

bool OCL2D_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data, *data_ref;
    setupBuffers(&data, NULL, &data_ref, n);
    {
        oclArgs argGPU, argCPU, argTranspose;
        checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data, FFT_FORWARD, n), "Create failed!");
        checkErr(runCombine2D(&argCPU, &argGPU, &argTranspose), "Run failed!");
        checkErr(oclRelease2D(NULL, data, &argCPU, &argGPU, &argTranspose), "Release failed!");
        write_normalized_image("OpenCL", "freq", data, n, true);
    }
    {
        oclArgs argGPU, argCPU, argTranspose;
        checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data, FFT_INVERSE, n), "Create failed!");
        checkErr(runCombine2D(&argCPU, &argGPU, &argTranspose), "Run failure!");
        checkErr(oclRelease2D(NULL, data, &argCPU, &argGPU, &argTranspose), "Release failed!");
        write_image("OpenCL", "spat-out", data, n);
    }
    return freeResults(&data, NULL, &data_ref, n) == 0;
}

double OCL2D_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[20];
    int minDim = n < TILE_DIM ? TILE_DIM * TILE_DIM : n * n;
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * minDim);

    oclArgs argGPU, argCPU, argTranspose;    
    checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data_in, FFT_FORWARD, n), "Create failed!");

    for (int i = 0; i < 20; ++i) {
        startTimer();
        runCombine2D(&argCPU, &argGPU, &argTranspose);
        measurements[i] = stopTimer();
    }

    checkErr(oclRelease2D(data_in, NULL, &argCPU, &argGPU, &argTranspose), "Release failed!");

    int res = freeResults(&data_in, NULL, NULL, n);
    return avg(measurements, 20);
}

//
// Algorithm
//

__inline cl_int runCombineHelper(oclArgs *argCPU, oclArgs *argGPU, cl_mem in, cl_mem out, int numBlocks, int syncLimit, bool dim2)
{
    int depth = log2_32(argCPU->n);
    const int lead = 32 - depth;
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = { (argCPU->dir == FFT_FORWARD ? 1.f : 1.f / argCPU->n), 0.f };
    const int nBlock = argCPU->n / numBlocks;
    const float w_angle = argCPU->dir * (M_2_PI / argCPU->n);
    const float w_bangle = argCPU->dir * (M_2_PI / nBlock);
    int bSize = (argCPU->n / 2);

    if (numBlocks > syncLimit) {

        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --depth;
        int steps = 0;
        int dist = bSize;
        oclSetKernelCPUArg(argCPU, in, out, w_angle, 0xFFFFFFFF << depth, steps, dist);
        checkErr(oclExecute(argCPU), "CPU Sync Kernel");
        // Instead of swapping input/output, run in place. The argGPU kernel needs to swap once.                
        while (--depth > breakSize) {
            dist >>= 1;
            ++steps;
            oclSetKernelCPUArg(argCPU, out, out, w_angle, 0xFFFFFFFF << depth, steps, dist);
            checkErr(oclExecute(argCPU), "CPU Sync Kernel");
        }
        ++depth;
        bSize = nBlock / 2;
        numBlocks = 1;        
        in = out;
    }

    // Calculate complete sequence in one launch and syncronize steps on GPU
    if (dim2)
        oclSetKernelGPU2DArg(argGPU, in, out, w_bangle, depth, scaleCpx, bSize);
    else
        oclSetKernelGPUArg(argGPU, w_angle, w_bangle, depth, lead, breakSize, scaleCpx, numBlocks, bSize);
    return oclExecute(argGPU);
}

__inline cl_int runCombine(oclArgs *argCPU, oclArgs *argGPU)
{
    int numBlocks = (int)(argCPU->global_work_size[0] / argCPU->local_work_size[0]);
    return runCombineHelper(argCPU, argGPU, argGPU->input, argGPU->output, numBlocks, HW_LIMIT, false);
}

__inline cl_int runCombine2D(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans)
{    
    cl_mem _in = argGPU->input;
    cl_mem _out = argGPU->output;
    // _in -> _out
    checkErr(runCombineHelper(argCPU, argGPU, _in, _out, argGPU->global_work_size[1], 1, true), "Helper 2D");
    oclSetKernelTransposeArg(argTrans, _out, _in);
    // _out -> _in
    checkErr(oclExecute(argTrans), "Transpose");    
    // _in -> _out
    checkErr(runCombineHelper(argCPU, argGPU, _in, _out, argGPU->global_work_size[1], 1, true), "Helper 2D 2");
    oclSetKernelTransposeArg(argTrans, _out, _in);
    // _out -> _in
    checkErr(oclExecute(argTrans), "Transpose 2");

    argCPU->input = argGPU->input = _out;
    argCPU->output = argGPU->output = _in;
    return CL_SUCCESS;
}
