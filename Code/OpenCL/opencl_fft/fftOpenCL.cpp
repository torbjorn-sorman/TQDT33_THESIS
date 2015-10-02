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
   return 1;
}

double PartSync_performance(const int n)
{
    return -1;
}

cl_int setupKernel(char *kernelFilename, char *kernelName, const int n, oclArgs *args)
{
    cl_int err = CL_SUCCESS;
     
    cl_platform_id platform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    char *kernelSource;
    
    // Get platform id
    if (err = clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS) return err;

    // Connect to a compute device    
    if (err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL) != CL_SUCCESS) return err;

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) return err;

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) return err;

    // Read kernel file as a char *
    std::string filename = kernelFilename;
    filename += ".cl";
    std::string data = getKernel(filename.c_str());

    char *src = (char *)malloc(sizeof(char) * (data.size() + 1));
    strcpy_s(src, sizeof(char) * (data.size() + 1), data.c_str());
    src[data.size()] = '\0';
    kernelSource = src;

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&src, NULL, &err);    
    if (err != CL_SUCCESS) return err;

    // Build the program executable
    if (err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        char *buffer = (char *)malloc(sizeof(char) * len);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("\n%s\n", buffer);
        free(buffer);
        return err;
    }

    // Create the compute kernel in the program we wish to run    
    kernel = clCreateKernel(program, kernelName, &err);
    if (err != CL_SUCCESS) return err;

    // If successful, store in the argument struct!
    args->n = n;
    args->platform = platform;
    args->device_id = device_id;
    args->context = context;
    args->commands = commands;
    args->program = program;
    args->kernel = kernel;
    args->kernelSource = kernelSource;
    return err;
}

cl_int setupDeviceMemoryData(oclArgs *args, cpx *dev_in)
{
    cl_int err = CL_SUCCESS;
    if (dev_in != NULL) {
        err = clEnqueueWriteBuffer(args->commands, args->input, CL_TRUE, 0, args->data_mem_size, dev_in, 0, NULL, NULL);
        if (err != CL_SUCCESS) return err;
        err = clFinish(args->commands);
    }
    return err;
}

#define HW_LIMIT (1024 / MAX_BLOCK_SIZE) * 7

cl_int setupWorkGroupsAndMemory(oclArgs *args)
{   
    cl_int err = CL_SUCCESS;
    const int n2 = args->n / 2;
    int grpDim = n2;
    int itmDim = n2 > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : n2;    
    int nBlock = args->n / itmDim;
    size_t data_mem_size = sizeof(cpx) * args->n;
    size_t shared_mem_size = sizeof(cpx) * itmDim * 2;
    size_t sync_mem_size = sizeof(int) * HW_LIMIT;
    cl_mem input = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS) return err;
    cl_mem output = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS) return err;
    cl_mem sync_in = clCreateBuffer(args->context, CL_MEM_READ_WRITE, sync_mem_size, NULL, &err);
    if (err != CL_SUCCESS) return err;
    cl_mem sync_out = clCreateBuffer(args->context, CL_MEM_READ_WRITE, sync_mem_size, NULL, &err);
    if (err != CL_SUCCESS) return err;
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &output);

    // If successful, store in the argument struct!
    args->global_work_size = grpDim;
    args->local_work_size = itmDim;
    args->shared_mem_size = shared_mem_size;
    args->data_mem_size = data_mem_size;
    args->nBlock = nBlock;
    args->input = input;
    args->output = output;
    args->sync_in = sync_in;
    args->sync_out = sync_out;

    return err;
}

void setKernelCPUArg(oclArgs *args, float w_angle, unsigned int lmask, unsigned int pmask, int steps, int dist)
{    
    clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &args->input);
    clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &args->output);
    clSetKernelArg(args->kernel, 2, sizeof(float), &w_angle);
    clSetKernelArg(args->kernel, 3, sizeof(unsigned int), &lmask);
    clSetKernelArg(args->kernel, 4, sizeof(unsigned int), &pmask);
    clSetKernelArg(args->kernel, 5, sizeof(int), &steps);
    clSetKernelArg(args->kernel, 6, sizeof(int), &dist);
}

void setKernelGPUArg(oclArgs *args, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, int n2)
{
    cl_int err = CL_SUCCESS;
    // (dev_buf *in, dev_buf *out, syn_buf *sync_in, syn_buf *sync_out, grp_buf *shared, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, const int n2)
    err = clSetKernelArg(args->kernel, 0, sizeof(cl_mem), &args->input);
    checkErr(err, err, "input");
    err = clSetKernelArg(args->kernel, 1, sizeof(cl_mem), &args->output);
    checkErr(err, err, "output");
    err = clSetKernelArg(args->kernel, 2, sizeof(cl_mem), &args->sync_in);
    checkErr(err, err, "sync_in");
    err = clSetKernelArg(args->kernel, 3, sizeof(cl_mem), &args->sync_out);
    checkErr(err, err, "sync_out");
    err = clSetKernelArg(args->kernel, 4, args->shared_mem_size, NULL);
    checkErr(err, err, "shared_mem_size");
    err = clSetKernelArg(args->kernel, 5, sizeof(float), &angle);
    checkErr(err, err, "angle");
    err = clSetKernelArg(args->kernel, 6, sizeof(float), &bAngle);
    checkErr(err, err, "bAngle");
    err = clSetKernelArg(args->kernel, 7, sizeof(int), &depth);
    checkErr(err, err, "depth");
    err = clSetKernelArg(args->kernel, 8, sizeof(int), &lead);
    checkErr(err, err, "lead");
    err = clSetKernelArg(args->kernel, 9, sizeof(int), &breakSize);
    checkErr(err, err, "breakSize");
    err = clSetKernelArg(args->kernel, 10, sizeof(cpx), &scale);
    checkErr(err, err, "scale");
    err = clSetKernelArg(args->kernel, 11, sizeof(int), &nBlocks);
    checkErr(err, err, "nBlocks");
    err = clSetKernelArg(args->kernel, 12, sizeof(int), &n2);
    checkErr(err, err, "n2");
}

cl_int runCombine(oclArgs *argCPU, oclArgs *argGPU)
{
    int depth = log2_32(argCPU->n);
    const int lead = 32 - depth;
    const int n2 = (argCPU->n / 2);
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = {(argCPU->dir == FFT_FORWARD ? 1.f : 1.f / argCPU->n), 0.f};    
    int numBlocks = argCPU->global_work_size / argCPU->local_work_size;
    const int nBlock = argCPU->n / argCPU->global_work_size;
    const float w_angle = argCPU->dir * (M_2_PI / argCPU->n);
    const float w_bangle = argCPU->dir * (M_2_PI / nBlock);
    int bSize = n2;

    if (argCPU->global_work_size >= HW_LIMIT) {

        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --depth;
        int steps = 0;
        int dist = n2;
        setKernelCPUArg(argCPU, w_angle, 0xFFFFFFFF << depth, (dist - 1) << steps, steps, dist);        
        oclExecute(argCPU);
        argGPU->output = argCPU->input;
        argGPU->input  = argCPU->output;
        argCPU->input  = argCPU->output;
        while (--depth > breakSize) {
            dist >>= 1;
            ++steps;
            setKernelCPUArg(argCPU, w_angle, 0xFFFFFFFF << depth, (dist - 1) << steps, steps, dist);
            oclExecute(argCPU);
        }
        ++depth;
        bSize = nBlock / 2;
        numBlocks = 1;
    }

    // Calculate complete sequence in one launch and syncronize on GPU
    setKernelGPUArg(argGPU, w_angle, w_bangle, depth, lead, breakSize, scaleCpx, numBlocks, bSize);
    return oclExecute(argGPU);
}


void oclRelease(cpx *dev_out, oclArgs *argCPU, oclArgs *argGPU, cl_int *error)
{
    cl_int err = CL_SUCCESS;
    if (dev_out != NULL) {
        err = clEnqueueReadBuffer(argGPU->commands, argGPU->output, CL_TRUE, 0, argGPU->data_mem_size, dev_out, 0, NULL, NULL);
        checkErr(err, "Read Buffer!");
    }
    *error = clFinish(argGPU->commands);
    free(argGPU->kernelSource);
    clReleaseMemObject(argGPU->input);
    clReleaseMemObject(argGPU->output);
    clReleaseMemObject(argGPU->sync_in);
    clReleaseMemObject(argGPU->sync_out);
    clReleaseProgram(argGPU->program);
    clReleaseProgram(argCPU->program);
    clReleaseKernel(argGPU->kernel);
    clReleaseKernel(argCPU->kernel);
    clReleaseCommandQueue(argGPU->commands);
    clReleaseCommandQueue(argCPU->commands);
    clReleaseContext(argGPU->context);
    clReleaseContext(argCPU->context);
}

int runPartialSync(const int n)
{
    oclArgs argCPU, argGPU;    
    cpx *data_in, *data_out, *data_ref;    
    setupTestData(&data_in, &data_out, &data_ref, &argCPU, FFT_FORWARD, n);
    argGPU.n = argCPU.n;
    argGPU.dir = argCPU.dir;

    setupKernel("kernelPartSync", "kernelCPU", n, &argCPU);
    setupKernel("kernelPartSync", "kernelGPU", n, &argGPU);
    
    setupWorkGroupsAndMemory(&argGPU);

    // Kernels must share some features!
    argCPU.global_work_size = argGPU.global_work_size;
    argCPU.local_work_size = argGPU.local_work_size;
    argCPU.input = argGPU.input;
    argCPU.output = argGPU.output;
    argCPU.data_mem_size = argGPU.data_mem_size;
    argCPU.nBlock = argGPU.nBlock;

    setupDeviceMemoryData(&argCPU, data_in);

    printf("Attempt to run with %d global and %d local items.\n", argGPU.global_work_size, argGPU.local_work_size);

    cl_int err = runCombine(&argCPU, &argGPU);

    checkErr(err, err, "Run failed!");
    
    oclRelease(data_out, &argCPU, &argGPU, &err);
    if (checkErr(err, err, "Validation failed!")) return err;    

    write_console(data_out, n);

    return freeResults(&data_in, &data_out, &data_ref, n);
}
