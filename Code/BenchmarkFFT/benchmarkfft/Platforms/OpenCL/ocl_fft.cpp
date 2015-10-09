#include "ocl_fft.h"

__inline void runCombine(oclArgs *argCPU, oclArgs *argGPU)
{
    int depth = log2_32(argCPU->n);
    const int lead = 32 - depth;
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = { (argCPU->dir == FFT_FORWARD ? 1.f : 1.f / argCPU->n), 0.f };
    int numBlocks = (int)(argCPU->global_work_size[0] / argCPU->local_work_size);
    const int nBlock = argCPU->n / numBlocks;
    const float w_angle = argCPU->dir * (M_2_PI / argCPU->n);
    const float w_bangle = argCPU->dir * (M_2_PI / nBlock);
    int bSize = (argCPU->n / 2);

    if (numBlocks >= HW_LIMIT) {

        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --depth;
        int steps = 0;
        int dist = bSize;
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

__inline void runCombine2D(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans)
{
    runCombine(argCPU, argGPU);
    oclSetKernelTransposeArg(argTrans);
    oclExecute(argTrans);

    runCombine(argCPU, argGPU);
    oclSetKernelTransposeArg(argTrans);
    oclExecute(argTrans);

    swap(&argGPU->input, &argGPU->output);
}

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

int OCL2D_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data_in = get_seq(n * n, 1);
    cpx *data_ref = get_seq(n * n, data_in);

    oclArgs argGPU, argCPU, argTranspose;
    err = oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data_in, FFT_FORWARD, n);
    checkErr(err, err, "Create failed!");
    runCombine2D(&argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Run failed!");
    err = oclRelease2D(data_in, &argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Release failed!");

    err = oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data_in, FFT_INVERSE, n);
    checkErr(err, err, "Create failed!");
    runCombine2D(&argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Run failed!");
    err = oclRelease2D(data_in, &argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Release failed!");

    return freeResults(&data_in, NULL, &data_ref, n) == 0;
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

double OCL2D_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[20];
    cpx *data_in = get_seq(n * n, 1);

    oclArgs argGPU, argCPU, argTranspose;
    err = oclCreateKernels2D(&argCPU, &argGPU, &argTranspose, data_in, FFT_FORWARD, n);
    checkErr(err, err, "Create failed!");

    for (int i = 0; i < 20; ++i) {
        startTimer();
        runCombine2D(&argCPU, &argGPU, &argTranspose);
        measurements[i] = stopTimer();
    }

    err = oclRelease2D(data_in, &argCPU, &argGPU, &argTranspose);
    checkErr(err, err, "Release failed!");

    int res = freeResults(&data_in, NULL, NULL, n);
    if (checkErr(err, err, "Validation failed!") || res)
        return -1;
    return avg(measurements, 20);
}