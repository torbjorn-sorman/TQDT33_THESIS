
#include "ocl_helper.h"

std::string getKernel(const char *filename)
{
    std::ifstream in(filename);
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return contents;
}

int checkErr(cl_int error, char *msg)
{
    if (error != CL_SUCCESS) {
        printf("Error Code: %d\nMessage: %s\n", error, msg);
        return 1;
    }
    return 0;
}

cl_int oclSetupKernel(oclArgs *args)
{
    cl_int err = CL_SUCCESS;
    cl_platform_id platform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    if (err = clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS)                               return err;
    if (err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL) != CL_SUCCESS)  return err;
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS)                                                                      return err;
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS)                                                                      return err;
    args->platform = platform;
    args->device_id = device_id;
    args->context = context;
    args->commands = commands;
    return err;
}

cl_int oclSetupProgram(char *kernelFilename, char *kernelName, oclArgs *args)
{
    cl_int err = CL_SUCCESS;
    cl_program program;
    cl_kernel kernel;
    char *kernelSource;
    // Read kernel file as a char *
    std::string filename = kernelFilename;
    filename = "Platforms/OpenCL/" + filename + ".cl";
    std::string data = getKernel(filename.c_str());

    char *src = (char *)malloc(sizeof(char) * (data.size() + 1));
    strcpy_s(src, sizeof(char) * (data.size() + 1), data.c_str());
    src[data.size()] = '\0';
    kernelSource = src;

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(args->context, 1, (const char **)&src, NULL, &err);
    if (err != CL_SUCCESS) return err;

    // Build the program executable
    if (err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, args->device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        char *buffer = (char *)malloc(sizeof(char) * len);
        clGetProgramBuildInfo(program, args->device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("\n%s\n", buffer);
        free(buffer);
        return err;
    }

    // Create the compute kernel in the program we wish to run    
    kernel = clCreateKernel(program, kernelName, &err);
    if (err != CL_SUCCESS) return err;

    args->program = program;
    args->kernel = kernel;
    args->kernelSource = kernelSource;
    return err;
}

cl_int oclSetupDeviceMemoryData(oclArgs *args, cpx *dev_in)
{
    cl_int err = CL_SUCCESS;
    if (dev_in != NULL) {
        err = clEnqueueWriteBuffer(args->commands, args->input, CL_TRUE, 0, args->data_mem_size, dev_in, 0, NULL, NULL);
        if (err != CL_SUCCESS) return err;
        err = clFinish(args->commands);
    }
    return err;
}

cl_int oclSetupWorkGroupsAndMemory(oclArgs *args)
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

    // If successful, store in the argument struct!
    args->global_work_size[0] = grpDim;
    args->global_work_size[1] = 1;
    args->global_work_size[2] = 1;
    args->local_work_size[0] = itmDim;
    args->local_work_size[1] = 1;
    args->local_work_size[2] = 1;
    args->shared_mem_size = shared_mem_size;
    args->data_mem_size = data_mem_size;
    args->nBlock = nBlock;
    args->input = input;
    args->output = output;
    args->sync_in = sync_in;
    args->sync_out = sync_out;

    return err;
}

cl_int oclSetupWorkGroupsAndMemory2D(oclArgs *args)
{
    cl_int err = CL_SUCCESS;
    const int n = args->n;
    int itmDim = (n / 2) > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : (n / 2);
    int nBlock = n / itmDim;
    int minSize = n < TILE_DIM ? TILE_DIM * TILE_DIM : n * n;
    size_t data_mem_size = sizeof(cpx) * minSize;
    size_t shared_mem_size = sizeof(cpx) * itmDim * 2;
    cl_mem input = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS)
        return err;
    cl_mem output = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS)
        return err;

    // If successful, store in the argument struct!
    args->global_work_size[0] = n * itmDim;
    args->global_work_size[1] = n / (itmDim * 2);
    args->global_work_size[2] = 1;
    args->local_work_size[0] = itmDim;
    args->local_work_size[1] = 1;
    args->local_work_size[2] = 1;
    args->shared_mem_size = shared_mem_size;
    args->data_mem_size = data_mem_size;
    args->nBlock = nBlock;
    args->input = input;
    args->output = output;

    return err;
}

cl_int oclCreateKernels(oclArgs *argCPU, oclArgs *argGPU, cpx *data_in, fftDir dir, const int n)
{
    argGPU->n = argCPU->n = n;
    argGPU->dir = argCPU->dir = dir;
    oclSetupKernel(argGPU);
    memcpy(argCPU, argGPU, sizeof(oclArgs));

    checkErr(oclSetupProgram("ocl_kernel", "kernelGPU", argGPU), "Failed to setup GPU Program!");
    checkErr(oclSetupProgram("ocl_kernel", "kernelCPU", argCPU), "Failed to setup CPU Program!");

    checkErr(oclSetupWorkGroupsAndMemory(argGPU), "Failed to setup GPU Program!");
    cl_int err = oclSetupDeviceMemoryData(argGPU, data_in);
    argCPU->global_work_size[0] = argGPU->global_work_size[0];
    argCPU->local_work_size[0] = argGPU->local_work_size[0];
    argCPU->input = argGPU->input;
    argCPU->output = argGPU->output;
    argCPU->data_mem_size = argGPU->data_mem_size;
    argCPU->nBlock = argGPU->nBlock;
    argCPU->workDim = 1;
    argGPU->workDim = 1;
    return err;
}

void setWorkDimForTranspose(oclArgs *argTranspose, const int n)
{
    int minDim = n > TILE_DIM ? (n / TILE_DIM) : 1;
    argTranspose->global_work_size[2] = 1;
    argTranspose->global_work_size[1] = minDim * THREAD_TILE_DIM;
    argTranspose->global_work_size[0] = minDim * THREAD_TILE_DIM;
    argTranspose->local_work_size[2] = 1;
    argTranspose->local_work_size[1] = THREAD_TILE_DIM;
    argTranspose->local_work_size[0] = THREAD_TILE_DIM;
    argTranspose->shared_mem_size = TILE_DIM * (TILE_DIM + 1) * sizeof(cpx);
    argTranspose->workDim = 2;
}

cl_int oclCreateKernels2D(oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans, cpx *data_in, fftDir dir, const int n)
{
    argGPU->n = argCPU->n = n;
    argGPU->dir = argCPU->dir = dir;
    oclSetupKernel(argGPU);
    memcpy(argCPU, argGPU, sizeof(oclArgs));
    memcpy(argTrans, argGPU, sizeof(oclArgs));

    checkErr(oclSetupProgram("ocl_kernel", "kernelGPU2D", argGPU), "Failed to setup GPU Program!");
    checkErr(oclSetupProgram("ocl_kernel", "kernelCPU2D", argCPU), "Failed to setup CPU Program!");
    checkErr(oclSetupProgram("ocl_kernel", "kernelTranspose", argTrans), "Failed to setup Transpose Program!");

    checkErr(oclSetupWorkGroupsAndMemory2D(argGPU), "Failed to setup Groups And Memory!");
    cl_int err = oclSetupDeviceMemoryData(argGPU, data_in);
    memcpy(argCPU->global_work_size, argGPU->global_work_size, sizeof(size_t) * 3);
    memcpy(argCPU->local_work_size, argGPU->local_work_size, sizeof(size_t) * 3);
    setWorkDimForTranspose(argTrans, n);
    argTrans->input = argCPU->input = argGPU->input;
    argTrans->output = argCPU->output = argGPU->output;
    argTrans->data_mem_size = argCPU->data_mem_size = argGPU->data_mem_size;
    argCPU->nBlock = argGPU->nBlock;
    argCPU->workDim = 2;
    argGPU->workDim = 2;
    return err;
}

cl_int oclRelease(cpx *dev_out, oclArgs *argCPU, oclArgs *argGPU)
{
    cl_int err = CL_SUCCESS;
    if (dev_out != NULL) {
        err = clEnqueueReadBuffer(argGPU->commands, argGPU->output, CL_TRUE, 0, argGPU->data_mem_size, dev_out, 0, NULL, NULL);
        checkErr(err, "Read Buffer!");
    }
    err = clFinish(argGPU->commands);
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
    clReleaseContext(argGPU->context);
    return err;
}

cl_int oclRelease2D(cpx *dev_in, cpx *dev_out, oclArgs *argCPU, oclArgs *argGPU, oclArgs *argTrans)
{
    cl_int err = CL_SUCCESS;
    if (dev_in != NULL) {    
        checkErr(clEnqueueReadBuffer(argGPU->commands, argGPU->input, CL_TRUE, 0, argGPU->data_mem_size, dev_in, 0, NULL, NULL), "Read Input Buffer!");
    }
    if (dev_out != NULL) {
        checkErr(clEnqueueReadBuffer(argGPU->commands, argGPU->output, CL_TRUE, 0, argGPU->data_mem_size, dev_out, 0, NULL, NULL), "Read Output Buffer!");
    }
    err = clFinish(argGPU->commands);
    free(argGPU->kernelSource);
    free(argTrans->kernelSource);
    checkErr(clReleaseMemObject(argGPU->input), "clReleaseMemObject->input");
    checkErr(clReleaseMemObject(argGPU->output), "clReleaseMemObject->output");
    checkErr(clReleaseProgram(argGPU->program), "clReleaseProgram->GPU");
    checkErr(clReleaseProgram(argCPU->program), "clReleaseProgram->CPU");
    checkErr(clReleaseProgram(argTrans->program), "clReleaseProgram->Trans");
    checkErr(clReleaseKernel(argGPU->kernel), "clReleaseKernel->GPU");
    checkErr(clReleaseKernel(argCPU->kernel), "clReleaseKernel->CPU");
    checkErr(clReleaseKernel(argTrans->kernel), "clReleaseKernel->Trans");
    checkErr(clReleaseCommandQueue(argGPU->commands), "clReleaseMemObject");
    err = clReleaseContext(argGPU->context);
    return err;
}

int freeResults(cpx **din, cpx **dout, cpx **dref, const int n)
{
    int res = 0;
    if (dref != NULL)   res = diff_seq(*din, *dref, n) > RELATIVE_ERROR_MARGIN;
    if (din != NULL)    free(*din);
    if (dout != NULL)   free(*dout);
    if (dref != NULL)   free(*dref);
    return res;
}

void setupBuffers(cpx **in, cpx **out, cpx **ref, const int n)
{
    char input_file[40];
    sprintf_s(input_file, 40, "Images/%u.ppm", n);
    int sz;
    size_t minSize = (n < TILE_DIM ? TILE_DIM * TILE_DIM : n * n) * sizeof(cpx);
    *in = (cpx *)malloc(minSize);
    if (out != NULL)
        *out = (cpx *)malloc(minSize);
    read_image(*in, input_file, &sz);
    *ref = (cpx *)malloc(minSize);
    memcpy(*ref, *in, minSize);
}