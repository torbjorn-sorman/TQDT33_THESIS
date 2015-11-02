#include "ocl_helper.h"
#include <iostream>

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
        getchar();
        exit(99);
    }
    return 0;
}

cl_int ocl_setup_kernels(oclArgs *args)
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
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
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

    // Read kernel file as a char *
    std::string filename = kernelFilename;
    filename = "Platforms/OpenCL/" + filename + ".cl";
    std::string data = getKernel(filename.c_str());

    char *src = (char *)malloc(sizeof(char) * (data.size() + 1));
    strcpy_s(src, sizeof(char) * (data.size() + 1), data.c_str());
    src[data.size()] = '\0';
    
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(args->context, 1, (const char **)&src, NULL, &err);
    if (err != CL_SUCCESS) return err;

    // Build the program executable
    if (err = clBuildProgram(program, 0, NULL, "-I G:/GitHub/TQDT33_THESIS/Code/BenchmarkFFT/benchmarkfft/Platforms", NULL, NULL) != CL_SUCCESS) {
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
    args->kernel_strings[0] = src;
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

cl_int oclSetupWorkGroupsAndMemory(oclArgs *args, int work_group_dim)
{
    cl_int err = CL_SUCCESS;
    const int n_half = args->n / 2;
    int grpDim = n_half;
    int itmDim = n_half > work_group_dim ? work_group_dim : n_half;
    size_t data_mem_size = sizeof(cpx) * args->n;
    args->input = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS) return err;
    args->output = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS) return err;

    // If successful, store in the argument struct!
    args->global_work_size[0] = grpDim;
    args->global_work_size[1] = 1;
    args->global_work_size[2] = 1;
    args->local_work_size[0] = itmDim;
    args->local_work_size[1] = 1;
    args->local_work_size[2] = 1;
    args->shared_mem_size = sizeof(cpx) * itmDim * 2;
    args->data_mem_size = data_mem_size;
    args->n_per_block = args->n / itmDim;

    return err;
}

cl_int oclSetupWorkGroupsAndMemory2D(oclArgs *args, int work_group_dim, int tile_dim)
{
    cl_int err = CL_SUCCESS;
    const int n = args->n;
    int itmDim = (n / 2) > work_group_dim ? work_group_dim : (n / 2);
    int n_per_block = n / itmDim;
    int minSize = n < tile_dim ? tile_dim * tile_dim : n * n;
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
    args->n_per_block = n_per_block;
    args->input = input;
    args->output = output;

    return err;
}

cl_int ocl_create_kernels(oclArgs *arg_cpu, oclArgs *arg_gpu, cpx *data_in, transform_direction dir, const int work_group_dim, const int n)
{
    arg_gpu->n = arg_cpu->n = n;
    arg_gpu->dir = arg_cpu->dir = dir;
    ocl_setup_kernels(arg_gpu);
    memcpy(arg_cpu, arg_gpu, sizeof(oclArgs));

    checkErr(oclSetupProgram("ocl_kernel", "ocl_kernel_local", arg_gpu), "Failed to setup GPU Program!");
    checkErr(oclSetupProgram("ocl_kernel", "ocl_kernel_global", arg_cpu), "Failed to setup CPU Program!");

    checkErr(oclSetupWorkGroupsAndMemory(arg_gpu, work_group_dim), "Failed to setup GPU Program!");
    cl_int err = CL_SUCCESS;
    if (data_in != NULL)
        err = oclSetupDeviceMemoryData(arg_gpu, data_in);
    arg_cpu->global_work_size[0] = arg_gpu->global_work_size[0];
    arg_cpu->local_work_size[0] = arg_gpu->local_work_size[0];
    arg_cpu->input = arg_gpu->input;
    arg_cpu->output = arg_gpu->output;
    arg_cpu->data_mem_size = arg_gpu->data_mem_size;
    arg_cpu->n_per_block = arg_gpu->n_per_block;
    arg_cpu->workDim = 1;
    arg_gpu->workDim = 1;
    return err;
}

cl_int ocl_create_timestamp_kernel(oclArgs *arg_target, oclArgs *arg_tm)
{
    memcpy(arg_tm, arg_target, sizeof(oclArgs));
    cl_int err = checkErr(oclSetupProgram("ocl_kernel", "ocl_timestamp_kernel", arg_tm), "Failed to setup GPU Program!");
    arg_tm->global_work_size[0] = 1;
    arg_tm->global_work_size[1] = 1;
    arg_tm->local_work_size[0] = 1;
    arg_tm->local_work_size[1] = 1;
    return err;
}

void setWorkDimForTranspose(oclArgs *argTranspose, int tile_dimension, int block_dimension, const int n)
{
    int minDim = n > tile_dimension ? (n / tile_dimension) : 1;
    argTranspose->global_work_size[2] = 1;
    argTranspose->global_work_size[1] = minDim * block_dimension;
    argTranspose->global_work_size[0] = minDim * block_dimension;
    argTranspose->local_work_size[2] = 1;
    argTranspose->local_work_size[1] = block_dimension;
    argTranspose->local_work_size[0] = block_dimension;
    argTranspose->shared_mem_size = tile_dimension * (tile_dimension + 1) * sizeof(cpx);
    argTranspose->workDim = 2;
}

cl_int oclCreateKernels2D(oclArgs *arg_cpu, oclArgs *arg_gpu, oclArgs *arg_transpose, cpx *data_in, transform_direction dir, const int work_group_dim, int tile_dimension, int block_dimension, const int n)
{
    arg_gpu->n = arg_cpu->n = n;
    arg_gpu->dir = arg_cpu->dir = dir;
    ocl_setup_kernels(arg_gpu);
    memcpy(arg_cpu, arg_gpu, sizeof(oclArgs));
    memcpy(arg_transpose, arg_gpu, sizeof(oclArgs));

    checkErr(oclSetupProgram("ocl_kernel", "ocl_kernel_local_row", arg_gpu), "Failed to setup GPU Program!");
    checkErr(oclSetupProgram("ocl_kernel", "ocl_kernel_global_row", arg_cpu), "Failed to setup CPU Program!");
    checkErr(oclSetupProgram("ocl_kernel", "ocl_transpose_kernel", arg_transpose), "Failed to setup Transpose Program!");

    checkErr(oclSetupWorkGroupsAndMemory2D(arg_gpu, work_group_dim, tile_dimension), "Failed to setup Groups And Memory!");
    cl_int err = oclSetupDeviceMemoryData(arg_gpu, data_in);
    memcpy(arg_cpu->global_work_size, arg_gpu->global_work_size, sizeof(size_t) * 3);
    memcpy(arg_cpu->local_work_size, arg_gpu->local_work_size, sizeof(size_t) * 3);
    setWorkDimForTranspose(arg_transpose, tile_dimension, block_dimension, n);
    arg_transpose->input = arg_cpu->input = arg_gpu->input;
    arg_transpose->output = arg_cpu->output = arg_gpu->output;
    arg_transpose->data_mem_size = arg_cpu->data_mem_size = arg_gpu->data_mem_size;
    arg_cpu->workDim = 2;
    arg_gpu->workDim = 2;
    return err;
}

cl_int oclRelease(cpx *dev_in, cpx *dev_out, oclArgs *arg_cpu, oclArgs *arg_gpu)
{
    cl_int err = CL_SUCCESS;
    if (dev_in != NULL) {
        err = clEnqueueReadBuffer(arg_gpu->commands, arg_gpu->input, CL_TRUE, 0, arg_gpu->data_mem_size, dev_in, 0, NULL, NULL);
        checkErr(err, "Read In Buffer!");
    }
    if (dev_out != NULL) {
        err = clEnqueueReadBuffer(arg_gpu->commands, arg_gpu->output, CL_TRUE, 0, arg_gpu->data_mem_size, dev_out, 0, NULL, NULL);
        checkErr(err, "Read Out Buffer!");
    }
    err = clFinish(arg_gpu->commands);
    free(arg_gpu->kernel_strings[0]);
    free(arg_cpu->kernel_strings[0]);
    clReleaseMemObject(arg_gpu->input);
    clReleaseMemObject(arg_gpu->output);
    clReleaseProgram(arg_gpu->program);
    clReleaseProgram(arg_cpu->program);
    clReleaseKernel(arg_gpu->kernel);
    clReleaseKernel(arg_cpu->kernel);
    clReleaseCommandQueue(arg_gpu->commands);
    clReleaseContext(arg_gpu->context);
    return err;
}

cl_int oclRelease2D(cpx *dev_in, cpx *dev_out, oclArgs *arg_cpu, oclArgs *arg_gpu, oclArgs *arg_transpose)
{
    cl_int err = CL_SUCCESS;
    if (dev_in != NULL) {    
        checkErr(clEnqueueReadBuffer(arg_gpu->commands, arg_gpu->input, CL_TRUE, 0, arg_gpu->data_mem_size, dev_in, 0, NULL, NULL), "Read Input Buffer!");
    }
    if (dev_out != NULL) {
        checkErr(clEnqueueReadBuffer(arg_gpu->commands, arg_gpu->output, CL_TRUE, 0, arg_gpu->data_mem_size, dev_out, 0, NULL, NULL), "Read Output Buffer!");
    }
    err = clFinish(arg_gpu->commands);
    free(arg_gpu->kernel_strings[0]);
    free(arg_cpu->kernel_strings[0]);    
    free(arg_transpose->kernel_strings[0]);    
    checkErr(clReleaseMemObject(arg_gpu->input), "clReleaseMemObject->input");
    checkErr(clReleaseMemObject(arg_gpu->output), "clReleaseMemObject->output");
    checkErr(clReleaseProgram(arg_gpu->program), "clReleaseProgram->GPU");
    checkErr(clReleaseProgram(arg_cpu->program), "clReleaseProgram->CPU");
    checkErr(clReleaseProgram(arg_transpose->program), "clReleaseProgram->Trans");
    checkErr(clReleaseKernel(arg_gpu->kernel), "clReleaseKernel->GPU");
    checkErr(clReleaseKernel(arg_cpu->kernel), "clReleaseKernel->CPU");
    checkErr(clReleaseKernel(arg_transpose->kernel), "clReleaseKernel->Trans");
    checkErr(clReleaseCommandQueue(arg_gpu->commands), "clReleaseMemObject");
    err = clReleaseContext(arg_gpu->context);
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

void setupBuffers(cpx **in, cpx **out, cpx **ref, int tile_dim, const int n)
{
    char input_file[40];
    sprintf_s(input_file, 40, "Images/%u.ppm", n);
    int sz;
    size_t minSize = (n < tile_dim ? tile_dim * tile_dim : n * n) * sizeof(cpx);
    *in = (cpx *)malloc(minSize);
    if (out != NULL)
        *out = (cpx *)malloc(minSize);
    read_image(*in, input_file, &sz);
    *ref = (cpx *)malloc(minSize);
    memcpy(*ref, *in, minSize);
}