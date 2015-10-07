
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
        printf("Error: %s\n", msg);
        return 1;
    }
    return 0;
}

int checkErr(cl_int error, cl_int args, char *msg)
{
    if (error != CL_SUCCESS) {
        printf("Error Code: %d\nMessage: %s\n", args, msg);
        return 1;
    }
    return 0;
}


cl_int oclSetup(char *kernelName, cpx *dev_in, oclArgs *args)
{
    cl_int err = CL_SUCCESS;
    if (err = clGetPlatformIDs(1, &args->platform, NULL) != CL_SUCCESS) return err;

    // Connect to a compute device    
    if (err = clGetDeviceIDs(args->platform, CL_DEVICE_TYPE_GPU, 1, &args->device_id, NULL) != CL_SUCCESS) return err;

    // Create a compute context
    args->context = clCreateContext(0, 1, &args->device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) return err;

    // Create a command commands
    args->commands = clCreateCommandQueue(args->context, args->device_id, 0, &err);
    if (err != CL_SUCCESS) return err;

    std::string filename = kernelName;
    filename += ".cl";
    std::string data = getKernel(filename.c_str());

    char *src = (char *)malloc(sizeof(char) * (data.size() + 1));
    strcpy_s(src, sizeof(char) * (data.size() + 1), data.c_str());
    src[data.size()] = '\0';

    // Create the compute program from the source buffer
    args->program = clCreateProgramWithSource(args->context, 1, (const char **)&src, NULL, &err);
    args->kernelSource = src;
    if (err != CL_SUCCESS) return err;

    // Build the program executable
    if (err = clBuildProgram(args->program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(args->program, args->device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        char *buffer = (char *)malloc(sizeof(char) * len);
        clGetProgramBuildInfo(args->program, args->device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("Failed to build program:\n%s\n", buffer);
        free(buffer);
        return err;
    }

    // Create the compute kernel in the program we wish to run    
    args->kernel = clCreateKernel(args->program, kernelName, &err);
    if (err != CL_SUCCESS) return err;

    const int n2 = args->n / 2;
    args->global_work_size = n2;
    args->local_work_size = (n2 > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : n2);

    args->nBlock = (int)(args->n / args->local_work_size);
    args->shared_mem_size = sizeof(cpx) * args->local_work_size * 2;

    size_t data_mem_size = sizeof(cpx) * args->n;
    args->input = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, NULL);
    args->output = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, NULL);

    if (dev_in != NULL) {
        err = clEnqueueWriteBuffer(args->commands, args->input, CL_TRUE, 0, data_mem_size, dev_in, 0, NULL, NULL);
        if (err != CL_SUCCESS) return err;
        err = clFinish(args->commands);
    }
    return err;
}


cl_int oclSetupKernel(const int n, oclArgs *args)
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

cl_int oclSetupWorkGroupsAndMemory(oclArgs *args, oclArgs *argsCrossGroups)
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

cl_int oclCreateKernels(oclArgs *argCPU, oclArgs *argGPU, cpx *data_in, fftDir dir, const int n)
{
    argGPU->n = argCPU->n = n;
    argGPU->dir = argCPU->dir = dir;
    oclSetupKernel(n, argGPU);
    memcpy(argCPU, argGPU, sizeof(oclArgs));
    cl_int err = oclSetupProgram("kernelOpenCL", "kernelGPU", argGPU);
    checkErr(err, err, "Failed to setup GPU Program!");
    err = oclSetupProgram("kernelOpenCL", "kernelCPU", argCPU);
    checkErr(err, err, "Failed to setup CPU Program!");
    checkErr(err, err, "Failed to setup GPU Program!");
    err = oclSetupWorkGroupsAndMemory(argGPU, argCPU);
    checkErr(err, err, "Failed to setup GPU Program!");
    err = oclSetupDeviceMemoryData(argGPU, data_in);
    argCPU->global_work_size = argGPU->global_work_size;
    argCPU->local_work_size = argGPU->local_work_size;
    argCPU->input = argGPU->input;
    argCPU->output = argGPU->output;
    argCPU->data_mem_size = argGPU->data_mem_size;
    argCPU->nBlock = argGPU->nBlock;
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

int freeResults(cpx **din, cpx **dout, cpx **dref, const int n)
{
    int res = 0;
    if (dref != NULL)   res = diff_seq(*din, *dref, n) > RELATIVE_ERROR_MARGIN;
    if (din != NULL)    free(*din);
    if (dout != NULL)   free(*dout);
    if (dref != NULL)   free(*dref);
    return res;
}