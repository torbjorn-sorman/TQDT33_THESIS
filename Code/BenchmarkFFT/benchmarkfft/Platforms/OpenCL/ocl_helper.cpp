#include "ocl_helper.h"

cl_int ocl_setup_work_groups(ocl_args *args);
cl_int ocl_setup_work_groups_2d(ocl_args *args);

cl_int ocl_setup_io_buffers(ocl_args *args, size_t data_mem_size);

int ocl_check_err(cl_int error, char *msg)
{
    if (error != CL_SUCCESS) {
        printf("Error Code: %d\nMessage: %s\n", error, msg);
        getchar();
        exit(99);
    }
    return 0;
}

cl_int ocl_setup_kernels(ocl_args *args, bool dim2)
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
    /*
    ocl_check_err(clGetPlatformIDs(1, &platform, NULL), "clGetPlatformIDs");
    ocl_check_err(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL), "clGetDeviceIDs");
    clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    ocl_check_err(err, "clCreateContext");
    clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    ocl_check_err(err, "clCreateCommandQueue");
    */
    args->platform = platform;
    args->device_id = device_id;
    args->context = context;
    args->commands = commands;
    if (dim2)
        return ocl_setup_work_groups_2d(args);
    ocl_check_err(ocl_setup_work_groups(args), "ocl_setup_work_groups");
    return ocl_setup_io_buffers(args, sizeof(cpx) * args->n);
}

std::wstring s2ws(const std::string& s)
{
    int len;
    int slength = (int)s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    wchar_t* buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
}

cl_int oclSetupProgram(char *kernelFilename, char *kernelName, ocl_args *args)
{
    cl_int err = CL_SUCCESS;
    cl_program program;
    cl_kernel kernel;

    // Read kernel file as a char *
    std::string filename = kernelFilename;
    filename = "Platforms/OpenCL/" + filename + ".cl";
    std::string data = get_kernel_src(s2ws(filename).c_str(), NULL);

    char *src = (char *)malloc(sizeof(char) * (data.size() + 1));
    strcpy_s(src, sizeof(char) * (data.size() + 1), data.c_str());
    src[data.size()] = '\0';
    
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
    args->kernel_strings[0] = src;
    return err;
}

cl_int oclSetupDeviceMemoryData(ocl_args *args, cpx *dev_in)
{
    cl_int err = CL_SUCCESS;
    if (dev_in != NULL) {
        err = clEnqueueWriteBuffer(args->commands, args->input, CL_TRUE, 0, args->data_mem_size, dev_in, 0, NULL, NULL);
        if (err != CL_SUCCESS) return err;
        err = clFinish(args->commands);
    }
    return err;
}

cl_int ocl_setup_work_groups(ocl_args *args)
{
    cl_int err = CL_SUCCESS;
    const int n_half = args->n / 2;
    int group_dim = n_half;
    int item_dim = n_half > OCL_GROUP_SIZE ? OCL_GROUP_SIZE : n_half;
    args->global_work_size[0] = group_dim;
    args->global_work_size[1] = 1;
    args->global_work_size[2] = 1;
    args->local_work_size[0] = item_dim;
    args->local_work_size[1] = 1;
    args->local_work_size[2] = 1;
    args->shared_mem_size = sizeof(cpx) * item_dim * 2;
    args->n_per_block = args->n / item_dim;
    args->number_of_blocks = group_dim / item_dim;
    return err;
}

cl_int ocl_setup_io_buffers(ocl_args *args, size_t data_mem_size)
{
    cl_int err = CL_SUCCESS;
    args->input = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS) 
        return err;
    args->output = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS) 
        return err;
    args->data_mem_size = data_mem_size;
    return err;
}

cl_int ocl_setup_work_groups_2d(ocl_args *args)
{
    cl_int err = CL_SUCCESS;
    const int n = args->n;
    int item_dim = (n / 2) > OCL_GROUP_SIZE ? OCL_GROUP_SIZE : (n / 2);
    int n_per_block = n / item_dim;
    int min_size = n < OCL_TILE_DIM ? OCL_TILE_DIM * OCL_TILE_DIM : n * n;
    size_t data_mem_size = sizeof(cpx) * min_size;
    size_t shared_mem_size = sizeof(cpx) * item_dim * 2;
    cl_mem input = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS)
        return err;
    cl_mem output = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS)
        return err;

    // If successful, store in the argument struct!
    args->global_work_size[0] = n * item_dim;
    args->global_work_size[1] = n / (item_dim * 2);
    args->global_work_size[2] = 1;
    args->local_work_size[0] = item_dim;
    args->local_work_size[1] = 1;
    args->local_work_size[2] = 1;
    args->shared_mem_size = shared_mem_size;
    args->data_mem_size = data_mem_size;
    args->n_per_block = n_per_block;
    args->input = input;
    args->output = output;
    args->number_of_blocks = n / (item_dim * 2);
    return err;
}

cl_int ocl_setup(ocl_args *a_host, ocl_args *a_dev, cpx *data_in, transform_direction dir, const int n)
{
    a_dev->n = a_host->n = n;
    a_dev->dir = a_host->dir = dir;
    ocl_setup_kernels(a_dev, false);
    memcpy(a_host, a_dev, sizeof(ocl_args));

    ocl_check_err(oclSetupProgram("ocl_kernel", "ocl_kernel_local", a_dev), "Failed to setup GPU Program!");
    ocl_check_err(oclSetupProgram("ocl_kernel", "ocl_kernel_global", a_host), "Failed to setup CPU Program!");
        
    cl_int err = CL_SUCCESS;
    if (data_in != NULL)
        err = oclSetupDeviceMemoryData(a_dev, data_in);
    a_host->workDim = a_dev->workDim = 1;
    return err;
}

cl_int ocl_setup_timestamp(ocl_args *arg_target, ocl_args *arg_tm)
{
    memcpy(arg_tm, arg_target, sizeof(ocl_args));
    cl_int err = ocl_check_err(oclSetupProgram("ocl_kernel", "ocl_timestamp_kernel", arg_tm), "Failed to setup GPU Program!");
    arg_tm->global_work_size[0] = 1;
    arg_tm->global_work_size[1] = 1;
    arg_tm->local_work_size[0] = 1;
    arg_tm->local_work_size[1] = 1;
    return err;
}

void setWorkDimForTranspose(ocl_args *argTranspose, const int n)
{
    int minDim = n > OCL_TILE_DIM ? (n / OCL_TILE_DIM) : 1;
    argTranspose->global_work_size[2] = 1;
    argTranspose->global_work_size[1] = minDim * OCL_BLOCK_DIM;
    argTranspose->global_work_size[0] = minDim * OCL_BLOCK_DIM;
    argTranspose->local_work_size[2] = 1;
    argTranspose->local_work_size[1] = OCL_BLOCK_DIM;
    argTranspose->local_work_size[0] = OCL_BLOCK_DIM;
    argTranspose->shared_mem_size = OCL_TILE_DIM * (OCL_TILE_DIM + 1) * sizeof(cpx);
    argTranspose->workDim = 2;
}

cl_int ocl_setup(ocl_args *a_host, ocl_args *a_dev, ocl_args *a_trans, cpx *data_in, transform_direction dir, const int n)
{
    a_dev->n = a_host->n = n;
    a_dev->dir = a_host->dir = dir;
    ocl_setup_kernels(a_dev, true);
    memcpy(a_host, a_dev, sizeof(ocl_args));
    memcpy(a_trans, a_dev, sizeof(ocl_args));

    ocl_check_err(oclSetupProgram("ocl_kernel", "ocl_kernel_local_row", a_dev), "Failed to setup GPU Program!");
    ocl_check_err(oclSetupProgram("ocl_kernel", "ocl_kernel_global_row", a_host), "Failed to setup CPU Program!");
    ocl_check_err(oclSetupProgram("ocl_kernel", "ocl_transpose_kernel", a_trans), "Failed to setup Transpose Program!");
        
    cl_int err = oclSetupDeviceMemoryData(a_dev, data_in);
    memcpy(a_host->global_work_size, a_dev->global_work_size, sizeof(size_t) * 3);
    memcpy(a_host->local_work_size, a_dev->local_work_size, sizeof(size_t) * 3);
    setWorkDimForTranspose(a_trans, n);
    a_trans->input = a_host->input = a_dev->input;
    a_trans->output = a_host->output = a_dev->output;
    a_trans->data_mem_size = a_host->data_mem_size = a_dev->data_mem_size;
    a_host->workDim = 2;
    a_dev->workDim = 2;
    return err;
}

cl_int ocl_shakedown(cpx *dev_in, cpx *dev_out, ocl_args *a_host, ocl_args *a_dev)
{
    cl_int err = CL_SUCCESS;
    if (dev_in != NULL) {
        err = clEnqueueReadBuffer(a_dev->commands, a_dev->input, CL_TRUE, 0, a_dev->data_mem_size, dev_in, 0, NULL, NULL);
        ocl_check_err(err, "Read In Buffer!");
    }
    if (dev_out != NULL) {
        err = clEnqueueReadBuffer(a_dev->commands, a_dev->output, CL_TRUE, 0, a_dev->data_mem_size, dev_out, 0, NULL, NULL);
        ocl_check_err(err, "Read Out Buffer!");
    }
    err = clFinish(a_dev->commands);
    free(a_dev->kernel_strings[0]);
    free(a_host->kernel_strings[0]);
    clReleaseMemObject(a_dev->input);
    clReleaseMemObject(a_dev->output);
    clReleaseProgram(a_dev->program);
    clReleaseProgram(a_host->program);
    clReleaseKernel(a_dev->kernel);
    clReleaseKernel(a_host->kernel);
    clReleaseCommandQueue(a_dev->commands);
    clReleaseContext(a_dev->context);
    return err;
}

cl_int ocl_shakedown(cpx *dev_in, cpx *dev_out, ocl_args *a_host, ocl_args *a_dev, ocl_args *a_trans)
{
    cl_int err = CL_SUCCESS;
    if (dev_in != NULL) {    
        ocl_check_err(clEnqueueReadBuffer(a_dev->commands, a_dev->input, CL_TRUE, 0, a_dev->data_mem_size, dev_in, 0, NULL, NULL), "Read Input Buffer!");
    }
    if (dev_out != NULL) {
        ocl_check_err(clEnqueueReadBuffer(a_dev->commands, a_dev->output, CL_TRUE, 0, a_dev->data_mem_size, dev_out, 0, NULL, NULL), "Read Output Buffer!");
    }
    err = clFinish(a_dev->commands);
    free(a_dev->kernel_strings[0]);
    free(a_host->kernel_strings[0]);    
    free(a_trans->kernel_strings[0]);    
    ocl_check_err(clReleaseMemObject(a_dev->input), "clReleaseMemObject->input");
    ocl_check_err(clReleaseMemObject(a_dev->output), "clReleaseMemObject->output");
    ocl_check_err(clReleaseProgram(a_dev->program), "clReleaseProgram->GPU");
    ocl_check_err(clReleaseProgram(a_host->program), "clReleaseProgram->CPU");
    ocl_check_err(clReleaseProgram(a_trans->program), "clReleaseProgram->Trans");
    ocl_check_err(clReleaseKernel(a_dev->kernel), "clReleaseKernel->GPU");
    ocl_check_err(clReleaseKernel(a_host->kernel), "clReleaseKernel->CPU");
    ocl_check_err(clReleaseKernel(a_trans->kernel), "clReleaseKernel->Trans");
    ocl_check_err(clReleaseCommandQueue(a_dev->commands), "clReleaseMemObject");
    err = clReleaseContext(a_dev->context);
    return err;
}

int ocl_free(cpx **din, cpx **dout, cpx **dref, const int n)
{
    int res = 0;
    if (dref != NULL)   res = diff_seq(*din, *dref, n) > RELATIVE_ERROR_MARGIN;
    if (din != NULL)    free(*din);
    if (dout != NULL)   free(*dout);
    if (dref != NULL)   free(*dref);
    return res;
}

void ocl_setup_buffers(cpx **in, cpx **out, cpx **ref, const int n)
{
    char input_file[40];
    sprintf_s(input_file, 40, "Images/%u.ppm", n);
    int sz;
    size_t minSize = (n < OCL_TILE_DIM ? OCL_TILE_DIM * OCL_TILE_DIM : n * n) * sizeof(cpx);
    *in = (cpx *)malloc(minSize);
    if (out != NULL)
        *out = (cpx *)malloc(minSize);
    read_image(*in, input_file, &sz);
    *ref = (cpx *)malloc(minSize);
    memcpy(*ref, *in, minSize);
}