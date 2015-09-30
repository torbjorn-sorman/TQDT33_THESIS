#include "fftGPUSync.h"

void GPUSync(fftDir dir, cpx *dev_in, cpx *dev_out, const int n);
cl_int oclSetArgs(oclArgs *args);

int GPUSync_validate(const int n)
{    
    cl_int err = CL_SUCCESS;
    cpx *data_in = get_seq(n, 1);
    cpx *data_out = get_seq(n);
    cpx *data_ref = get_seq(n, data_in);
    oclArgs arguments;

    arguments.n = n;
    arguments.dir = FFT_FORWARD;    
    oclSetup("kernelGPUSync", data_in, &arguments);
    oclSetArgs(&arguments);
    oclExecute(&arguments);
    oclRelease(data_out, &arguments, &err);
    if (checkErr(err, err, "Validation failed!"))
        return err;

    arguments.n = n;
    arguments.dir = FFT_INVERSE;
    oclSetup("kernelGPUSync", data_out, &arguments);
    oclSetArgs(&arguments);
    oclExecute(&arguments);
    oclRelease(data_in, &arguments, &err);
    if (checkErr(err, err, "Validation failed!"))
        return err;
    
    int res = checkError(data_in, data_ref, n, 1);
    free(data_in);
    free(data_out);
    free(data_ref);
    return res;
}

double GPUSync_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[20];
    cpx *data_in = get_seq(n, 1);
    cpx *data_out = get_seq(n);
    oclArgs arguments;
    arguments.n = n;
    arguments.dir = FFT_FORWARD;
    oclSetup("kernelGPUSync", data_in, &arguments);
    oclSetArgs(&arguments);

    for (int i = 0; i < 20; ++i) {
        startTimer();
        oclExecute(&arguments);
        measurements[i] = stopTimer();
    }

    oclRelease(data_out, &arguments, &err);
    if (checkErr(err, err, "Validation failed!"))
        return -1;
    return avg(measurements, 20);
}

cl_int oclSetArgs(oclArgs *args)
{
    const float w_angle = args->dir * (M_2_PI / args->n);
    const float w_bangle = args->dir * (M_2_PI / args->nBlock);
    const cpx scale = { (args->dir == FFT_FORWARD ? 1.f : 1.f / args->n), 0.f };
    const int depth = log2_32(args->n);
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    const int n2 = args->n / 2;
    cl_int err = 0;
    int arg = 0;
    err = clSetKernelArg(args->kernel, arg++, args->shared_mem_size, NULL);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(cl_mem), &args->input);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(cl_mem), &args->output);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(cl_mem), &args->sync_in);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(cl_mem), &args->sync_out);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(float), &w_angle);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(float), &w_bangle);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(int), &depth);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(int), &breakSize);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(cpx), &scale);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(int), &args->nBlock);
    err |= clSetKernelArg(args->kernel, arg++, sizeof(int), &n2);
    return err;
}