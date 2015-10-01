#include "fftOpenCL.h"

void runGPUSync(oclArgs *args);
void runPartSync(oclArgs *args);

void setupTestData(cpx **din, cpx **dout, cpx **dref, oclArgs *args, fftDir dir, const int n)
{
    if (din != NULL)
        *din = get_seq(n, 1);
    if (dout != NULL)
        *dout = get_seq(n);
    if (dref != NULL)
        *dref = get_seq(n, *din);
    args->n = n;
    args->dir = FFT_FORWARD;
}

int freeResults(cpx **din, cpx **dout, cpx **dref, const int n)
{
    int res = 0;
    if (dref != NULL)
        res = checkError(*din, *dref, n, 1);
    if (din != NULL)
        free(*din);
    if (dout != NULL)
        free(*dout);
    if (dref != NULL)
        free(*dref);
    return res;
}

int GPUSync_validate(const int n)
{    
    if (n > (MAX_BLOCK_SIZE * 2)) return 0;
    cl_int err = CL_SUCCESS;
    cpx *data_in, *data_out, *data_ref;
    oclArgs arguments;
    setupTestData(&data_in, &data_out, &data_ref, &arguments, FFT_FORWARD, n);
    oclSetup("kernelGPUSync", data_in, &arguments);
    runGPUSync(&arguments);
    oclRelease(data_out, &arguments, &err);
    if (checkErr(err, err, "Validation failed!")) return err;

    arguments.dir = FFT_INVERSE;
    swap(&arguments.input, &arguments.output);
    oclSetup("kernelGPUSync", data_out, &arguments);
    oclRelease(data_in, &arguments, &err);
    if (checkErr(err, err, "Validation failed!")) return err;
    
    return freeResults(&data_in, &data_out, &data_ref, n);
}

double GPUSync_performance(const int n)
{
    if (n > (MAX_BLOCK_SIZE * 2))
        return -1.0;
    cl_int err = CL_SUCCESS;
    double measurements[20];
    cpx *data_in, *data_out;
    oclArgs arguments;
    setupTestData(&data_in, &data_out, NULL, &arguments, FFT_FORWARD, n);
    oclSetup("kernelGPUSync", NULL, &arguments);    
    for (int i = 0; i < 20; ++i) {
        startTimer();
        runGPUSync(&arguments);
        measurements[i] = stopTimer();
    }
    oclRelease(data_out, &arguments, &err);
    int res = freeResults(&data_in, &data_out, NULL, n);
    if (checkErr(err, err, "Validation failed!") || res)
        return -1;
    return avg(measurements, 20);
}

void runGPUSync(oclArgs *args)
{
    const float w_angle = args->dir * (M_2_PI / args->n);    
    const cpx scale = { (args->dir == FFT_FORWARD ? 1.f : 1.f / args->n), 0.f };
    const int depth = log2_32(args->n);
    const int lead = 32 - depth;
    const int n2 = args->n / 2;    
    int arg = 0;
    clSetKernelArg(args->kernel, arg++, args->shared_mem_size, NULL);
    clSetKernelArg(args->kernel, arg++, sizeof(cl_mem), &args->input);
    clSetKernelArg(args->kernel, arg++, sizeof(cl_mem), &args->output);    
    clSetKernelArg(args->kernel, arg++, sizeof(float), &w_angle);
    clSetKernelArg(args->kernel, arg++, sizeof(int), &depth);
    clSetKernelArg(args->kernel, arg++, sizeof(int), &lead);
    clSetKernelArg(args->kernel, arg++, sizeof(cpx), &scale);
    clSetKernelArg(args->kernel, arg++, sizeof(int), &n2);
    oclExecute(args);
}

int PartSync_validate(const int n)
{
    cl_int err = CL_SUCCESS;
    cpx *data_in, *data_out, *data_ref;
    oclArgs arguments;
    setupTestData(&data_in, &data_out, &data_ref, &arguments, FFT_FORWARD, n);
    oclSetup("kernelPartSync", data_in, &arguments);
    
    runPartSync(&arguments);
    oclRelease(data_out, &arguments, &err);
    if (checkErr(err, err, "Validation failed!")) return err;

    arguments.dir = FFT_INVERSE;
    swap(&arguments.input, &arguments.output);
    oclSetup("kernelPartSync", data_in, &arguments);
    runPartSync(&arguments);
    oclRelease(data_in, &arguments, &err);
    if (checkErr(err, err, "Validation failed!")) return err;

    return freeResults(&data_in, &data_out, &data_ref, n);
}

double PartSync_performance(const int n)
{
    cl_int err = CL_SUCCESS;
    double measurements[20];
    cpx *data_in, *data_out;
    oclArgs arguments;
    setupTestData(&data_in, &data_out, NULL, &arguments, FFT_FORWARD, n);
    oclSetup("kernelPartSync", NULL, &arguments);
    for (int i = 0; i < 20; ++i) {
        startTimer();
        runPartSync(&arguments);
        measurements[i] = stopTimer();
    }
    oclRelease(data_out, &arguments, &err);
    int res = freeResults(&data_in, &data_out, NULL, n);
    if (checkErr(err, err, "Validation failed!") || res) 
        return -1;
    return avg(measurements, 20);
}

void runPartSync(oclArgs *args)
{
    const float w_angle = args->dir * (M_2_PI / args->n);
    const float w_bangle = args->dir * (M_2_PI / args->nBlock);
    const cpx scaleCpx = { (args->dir == FFT_FORWARD ? 1.f : 1.f / args->n), 0.f };
    const int depth = log2_32(args->n);
    const int lead = 32 - depth;
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    const int n2 = args->n / 2;
    const int blocks = args->global_work_size[0] / args->local_work_size[0];

    int steps = 0;
    int dist = n2;
    int bit = depth - 1;
    unsigned int lmask = 0xFFFFFFFF << bit;
    unsigned int pmask = (dist - 1) << steps;

    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &args->input);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &args->output);
    clSetKernelArg(args->kernel, 2, sizeof(float), &w_angle);       

    if (blocks > 1) {
        clSetKernelArg(args->kernel, 3, sizeof(unsigned int), &lmask);
        clSetKernelArg(args->kernel, 4, sizeof(unsigned int), &pmask);
        clSetKernelArg(args->kernel, 5, sizeof(int), &steps);
        clSetKernelArg(args->kernel, 6, sizeof(int), &dist);
        oclExecute(args);
        while (--bit > breakSize) {
            dist >>= 1;
            ++steps;
            clSetKernelArg(args->kernel, 3, sizeof(unsigned int), &lmask);
            clSetKernelArg(args->kernel, 4, sizeof(unsigned int), &pmask);
            clSetKernelArg(args->kernel, 5, sizeof(int), &steps);
            clSetKernelArg(args->kernel, 6, sizeof(int), &dist);
            oclExecute(args);
        }
        int nextBit = bit + 1;
        int nBlocks2 = args->nBlock >> 1;
        clSetKernelArg(args->kernel, 3, sizeof(cpx), &scaleCpx);
        clSetKernelArg(args->kernel, 4, sizeof(int), &nextBit);
        clSetKernelArg(args->kernel, 5, sizeof(int), &lead);
        clSetKernelArg(args->kernel, 6, sizeof(int), &nBlocks2);
        oclExecute(args);
        cl_mem tmp = args->input;
        args->input = args->output;
        args->output = tmp;
    }
    else {
        oclExecute(args);
    }
}