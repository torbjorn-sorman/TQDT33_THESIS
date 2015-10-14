#include "ocl_fft.h"

__inline void runCombine(oclArgs *argCPU, oclArgs *argGPU);
__inline void runCombine2D(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans);

//
// 1D
//

int OCL_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data_in = get_seq(n, 1);
    cpx *data_out = get_seq(n);
    cpx *data_ref = get_seq(n, data_in);

    oclArgs argGPU, argCPU;
    err = oclCreateKernels(&argCPU, &argGPU, data_in, FFT_FORWARD, n);
    checkErr(err, err, "Create failed!");
    runCombine(&argCPU, &argGPU);
    checkErr(err, err, "Run failed!");
    err = oclRelease(data_out, &argCPU, &argGPU);
    checkErr(err, err, "Release failed!");

    err = oclCreateKernels(&argCPU, &argGPU, data_out, FFT_INVERSE, n);
    checkErr(err, err, "Create failed!");
    runCombine(&argCPU, &argGPU);
    checkErr(err, err, "Run failed!");
    err = oclRelease(data_in, &argCPU, &argGPU);
    checkErr(err, err, "Release failed!");

    return freeResults(&data_in, &data_out, &data_ref, n) == 0;
}

double OCL_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[20];
    cpx *data_in = get_seq(n, 1);

    oclArgs argGPU, argCPU;
    err = oclCreateKernels(&argCPU, &argGPU, data_in, FFT_FORWARD, n);
    checkErr(err, err, "Create failed!");

    for (int i = 0; i < 20; ++i) {
        startTimer();
        runCombine(&argCPU, &argGPU);
        measurements[i] = stopTimer();
    }

    err = oclRelease(data_in, &argCPU, &argGPU);
    checkErr(err, err, "Release failed!");

    int res = freeResults(&data_in, NULL, NULL, n);
    if (checkErr(err, err, "Validation failed!") || res)
        return -1;
    return avg(measurements, 20);
}

//
// 2D
//

int OCL2D_validate(const int n)
{
    cl_int err = CL_SUCCESS;
<<<<<<< HEAD
    cpx *data, *data_ref;
    setupBuffers(&data, NULL, &data_ref, n);
    {
        oclArgs argGPU, argCPU, argTranspose;
        checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data, FFT_FORWARD, n), "Create failed!");
        checkErr(runCombine2D(&argCPU, &argGPU, &argTranspose), "Run failed!");
        checkErr(oclRelease2D(NULL, data, &argCPU, &argGPU, &argTranspose), "Release failed!");
        write_normalized_image("OpenCL", "freq", data, n, true);
        //write_image("OpenCL", "freq", data, n);
    }
    {
        oclArgs argGPU, argCPU, argTranspose;
        checkErr(oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data, FFT_INVERSE, n), "Create failed!");
        checkErr(runCombine2D(&argCPU, &argGPU, &argTranspose), "Run failure!");
        checkErr(oclRelease2D(NULL, data, &argCPU, &argGPU, &argTranspose), "Release failed!");
        write_image("OpenCL", "spat-out", data, n);
    }
    return freeResults(&data, NULL, &data_ref, n) == 0;
=======
    cpx *data_in, *data_out, *data_ref;
    setupBuffers(&data_in, &data_out, &data_ref, n);

    oclArgs argGPU, argCPU, argTranspose;
    err = oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data_in, FFT_FORWARD, n);
    checkErr(err, err, "Create failed!");
    runCombine2D(&argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Run failed!");
    err = oclRelease2D(data_in, data_out, &argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Release failed!");
    //write_normalized_image("OpenCL", "freq", data_in, n, true);
    write_image("OpenCL", "freq-in", data_in, n);
    write_image("OpenCL", "freq-out", data_out, n);

    err = oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data_in, FFT_INVERSE, n);
    checkErr(err, err, "Create failed!");
    runCombine2D(&argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Run failed!");
    err = oclRelease2D(data_in, data_out, &argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Release failed!");
    write_image("OpenCL", "spa-in", data_in, n);
    write_image("OpenCL", "spat-out", data_out, n);

    return freeResults(&data_in, &data_out, &data_ref, n) == 0;
>>>>>>> parent of f544fdb... blaj
}

double OCL2D_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[20];
    int minDim = n < TILE_DIM ? TILE_DIM * TILE_DIM : n * n;
    cpx *data_in = (cpx *)malloc(sizeof(cpx) * minDim);

    oclArgs argGPU, argCPU, argTranspose;
    err = oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data_in, FFT_FORWARD, n);
    checkErr(err, err, "Create failed!");

    for (int i = 0; i < 20; ++i) {
        startTimer();
        runCombine2D(&argCPU, &argGPU, &argTranspose);
        measurements[i] = stopTimer();
    }

    err = oclRelease2D(data_in, NULL, &argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Release failed!");

    int res = freeResults(&data_in, NULL, NULL, n);
    if (checkErr(err, err, "Validation failed!") || res)
        return -1;
    return avg(measurements, 20);
}

//
// Algorithm
//

__inline void runCombineHelper(oclArgs *argCPU, oclArgs *argGPU, int numBlocks)
{
    int depth = log2_32(argCPU->n);
    const int lead = 32 - depth;
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = { (argCPU->dir == FFT_FORWARD ? 1.f : 1.f / argCPU->n), 0.f };

    int numBlocks = (int)(argCPU->global_work_size[0] / argCPU->local_work_size[0]);
    const int nBlock = argCPU->n / numBlocks;
    const float w_angle = argCPU->dir * (M_2_PI / argCPU->n);
    const float w_bangle = argCPU->dir * (M_2_PI / nBlock);
    int bSize = (argCPU->n / 2);

    if (numBlocks >= HW_LIMIT) {

        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --depth;
        int steps = 0;
        int dist = bSize;
<<<<<<< HEAD
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
=======
        oclSetKernelCPUArg(argCPU, w_angle, 0xFFFFFFFF << depth, steps, dist);
        oclExecute(argCPU);
        // Instead of swapping input/output, run in place. The argGPU kernel needs to swap once.
        swap(&argGPU->input, &argGPU->output);
        argCPU->input = argCPU->output;
        while (--depth > breakSize) {
            dist >>= 1;
            ++steps;
            oclSetKernelCPUArg(argCPU, w_angle, 0xFFFFFFFF << depth, steps, dist);
            oclExecute(argCPU);
        }
        ++depth;
        bSize = nBlock / 2;
        numBlocks = 1;
    }

    // Calculate complete sequence in one launch and syncronize steps on GPU
    oclSetKernelGPUArg(argGPU, w_angle, w_bangle, depth, lead, breakSize, scaleCpx, numBlocks, bSize);
    oclExecute(argGPU);
}

__inline void runCombine2D(oclArgs *argCPU, oclArgs *argGPU)
{
    int depth = log2_32(argCPU->n);
    const int lead = 32 - depth;
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = { (argCPU->dir == FFT_FORWARD ? 1.f : 1.f / argCPU->n), 0.f };

    const int nBlock = argCPU->n / argCPU->global_work_size[1];
    const float w_angle = argCPU->dir * (M_2_PI / argCPU->n);
    const float w_bangle = argCPU->dir * (M_2_PI / nBlock);
    int bSize = argCPU->n;

    if (argCPU->global_work_size[1] > 1) {

        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --depth;
        int steps = 0;
        int dist = argCPU->n / 2;
        oclSetKernelCPUArg(argCPU, w_angle, 0xFFFFFFFF << depth, steps, dist);
        oclExecute(argCPU);

        // Instead of swapping input/output, run in place. The argGPU kernel needs to swap once.        
        swap(&argGPU->input, &argGPU->output);
        argCPU->input = argCPU->output;
        while (--depth > breakSize) {
            dist >>= 1;
            ++steps;
            oclSetKernelCPUArg(argCPU, w_angle, 0xFFFFFFFF << depth, steps, dist);
            oclExecute(argCPU);
        }
        ++depth;
        bSize = nBlock;
>>>>>>> parent of f544fdb... blaj
    }

    // Calculate complete sequence in one launch and syncronize steps on GPU
    oclSetKernelGPUArg(argGPU, w_angle, w_bangle, depth, lead, breakSize, scaleCpx, 1, bSize);
    oclExecute(argGPU);
}

__inline void runCombine(oclArgs *argCPU, oclArgs *argGPU)
{
    int numBlocks = (int)(argCPU->global_work_size[0] / argCPU->local_work_size[0]);
    runCombineHelper(argCPU, argGPU, numBlocks);
}

__inline void runCombine2D(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans)
{    
<<<<<<< HEAD
    cl_mem _in = argGPU->input;
    cl_mem _out = argGPU->output;    
    // _in -> _out
    checkErr(runCombineHelper(argCPU, argGPU, _in, _out, argGPU->global_work_size[1], 1, true), "Helper 2D");
    // _out -> _in
    oclSetKernelTransposeArg(argTrans, _out, _in);    
    checkErr(oclExecute(argTrans), "Transpose");    
    // _in -> _out    
    checkErr(runCombineHelper(argCPU, argGPU, _in, _out, argGPU->global_work_size[1], 1, true), "Helper 2D 2");
    // _out -> _in
    oclSetKernelTransposeArg(argTrans, _out, _in);    
    checkErr(oclExecute(argTrans), "Transpose 2");

    argCPU->input = argGPU->input = _out;
    argCPU->output = argGPU->output = _in;
    return CL_SUCCESS;
=======
    /*
    runCombine2D(argCPU, argGPU);
    argCPU->input = argGPU->input;
    argCPU->output = argGPU->output;
    */
    oclSetKernelTransposeArg(argTrans);
    oclExecute(argTrans);
    /*
    runCombine(argCPU, argGPU);
    argCPU->input = argGPU->input;
    argCPU->output = argGPU->output;
    
    oclSetKernelTransposeArg(argTrans);
    oclExecute(argTrans);

    swap(&argGPU->input, &argGPU->output);
    */
>>>>>>> parent of f544fdb... blaj
}
