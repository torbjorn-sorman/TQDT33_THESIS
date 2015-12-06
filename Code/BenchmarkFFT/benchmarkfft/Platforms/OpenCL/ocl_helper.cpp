#include "ocl_helper.h"

#include <iostream>

cl_int ocl_setup_io_buffers(ocl_args *args, size_t data_mem_size);

int ocl_check_err(cl_int error, char *msg)
{
    if (error != CL_SUCCESS) {
        printf("Error Code: %d\nMessage: %s\n", error, msg);
#pragma warning(suppress: 6031)
        getchar();
        exit(99);
    }
    return 0;
}

cl_int ocl_setup_work_groups(ocl_args *args, const int group_size)
{
    cl_int err = CL_SUCCESS;
    const int n_half = args->n / 2;
    int group_dim = n_half;
    int item_dim = n_half > group_size ? group_size : n_half;
    args->work_size[0] = batch_count(args->n);
    args->work_size[1] = group_dim;
    args->work_size[2] = 1;
    args->group_work_size[0] = 1;
    args->group_work_size[1] = item_dim;
    args->group_work_size[2] = 1;
    args->number_of_blocks = group_dim / item_dim;
    return err;
}

cl_int ocl_setup_work_groups_2d(ocl_args *args, const int group_size)
{
    cl_int err = CL_SUCCESS;
    const int n = args->n;
    const int n_half = n >> 1;
    int item_dim = n_half > group_size ? group_size : n_half;
    int n_per_block = n / item_dim;
    args->work_size[0] = n * item_dim;
    args->work_size[1] = n_half / item_dim;
    args->work_size[2] = batch_count(args->n);
    args->group_work_size[0] = item_dim;
    args->group_work_size[1] = 1;
    args->group_work_size[2] = 1;
    args->number_of_blocks = (int)args->work_size[1];
    return err;
}

void setWorkDimForTranspose(ocl_args *args, const int tile_dim, const int block_dim, const int n)
{
    int minDim = n > tile_dim ? (n / tile_dim) : 1;
    args->work_size[0] = minDim * block_dim;
    args->work_size[1] = minDim * block_dim;
    args->work_size[2] = batch_count(args->n);
    args->group_work_size[0] = block_dim;
    args->group_work_size[1] = block_dim;
    args->group_work_size[2] = 1;
}

cl_int ocl_setup_kernels(ocl_args *args, const int group_size, bool dim2)
{
    cl_int err = CL_SUCCESS;
    ocl_check_err(ocl_get_platform(&args->platform), "ocl_get_platform");
    ocl_check_err(clGetDeviceIDs(args->platform, CL_DEVICE_TYPE_GPU, 1, &args->device_id, NULL), "clGetDeviceIDs");
    args->context = clCreateContext(0, 1, &args->device_id, NULL, NULL, &err);
    ocl_check_err(err, "clCreateContext");
    args->commands = clCreateCommandQueue(args->context, args->device_id, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    ocl_check_err(err, "clCreateCommandQueue");

    size_t data_size = sizeof(cpx) * batch_size(args->n);
    if (dim2) {
        data_size = sizeof(cpx) * batch_size(args->n * args->n);
        ocl_check_err(ocl_setup_work_groups_2d(args, group_size), "ocl_setup_work_groups_2d");
    }
    else {
        ocl_check_err(ocl_setup_work_groups(args, group_size), "ocl_setup_work_groups");
    }
    return ocl_setup_io_buffers(args, data_size);
}

cl_int ocl_setup_program(std::string kernel_filename, char *kernel_name, ocl_args *args, int tile_dim, int block_dim)
{
    cl_int err = CL_SUCCESS;
    cl_program program;
    
    std::string str = get_file_content(std::string("Platforms/OpenCL/" + kernel_filename + ".cl"));
    
    manip_content(&str, L"OCL_TILE_DIM", tile_dim);
    manip_content(&str, L"OCL_BLOCK_DIM", block_dim);
#if defined(OCL_REF)
    manip_content(&str, L"SHOW_REF", 1);
#endif
    char *src = get_kernel_src_from_string(str, NULL);

    program = clCreateProgramWithSource(args->context, 1, (const char **)&src, NULL, &err);
    if (err != CL_SUCCESS) return err;

    if (err = clBuildProgram(program, 0, NULL, "-cl-single-precision-constant -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, args->device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        char *buffer = (char *)malloc(sizeof(char) * len);
        clGetProgramBuildInfo(program, args->device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("\n%s\n", buffer);
        free(buffer);
        return err;
    }
    
    args->program = program;
    args->kernel = clCreateKernel(program, kernel_name, &err);
    free(src);
    return err;
}

cl_int ocl_setup_program(std::string kernel_filename, char *kernel_name, ocl_args *args)
{
    return ocl_setup_program(kernel_filename, kernel_name, args, 0, 0);
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

cl_int ocl_setup_io_buffers(ocl_args *args, size_t data_mem_size)
{    
    cl_int err = CL_SUCCESS;
    args->input = clCreateBuffer(args->context, CL_MEM_READ_WRITE, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS)
        return err;
    args->output = clCreateBuffer(args->context, CL_MEM_WRITE_ONLY, data_mem_size, NULL, &err);
    if (err != CL_SUCCESS)
        return err;
    args->data_mem_size = data_mem_size;
    return err;
}

cl_int ocl_setup(ocl_args *a_host, ocl_args *a_dev, cpx *data_in, transform_direction dir, const int group_size, const int n)
{
    a_dev->n = a_host->n = n;
    a_dev->dir = a_host->dir = dir;
    ocl_setup_kernels(a_dev, group_size, false);
    memcpy(a_host, a_dev, sizeof(ocl_args));

    ocl_check_err(ocl_setup_program("ocl_kernel", "ocl_kernel_local", a_dev, 0, 0), "Failed to setup GPU Program 1D!");
    ocl_check_err(ocl_setup_program("ocl_kernel", "ocl_kernel_global", a_host, 0, 0), "Failed to setup CPU Program!");

    cl_int err = CL_SUCCESS;
    if (data_in != NULL)
        err = oclSetupDeviceMemoryData(a_dev, data_in);
    a_host->workDim = a_dev->workDim = 2;
    return err;
}

cl_int ocl_setup_timestamp(ocl_args *arg_target, ocl_args *arg_tm)
{
    memcpy(arg_tm, arg_target, sizeof(ocl_args));
    cl_int err = ocl_check_err(ocl_setup_program("ocl_kernel", "ocl_timestamp_kernel", arg_tm), "Failed to setup GPU Program!");    
    arg_tm->work_size[0] = 1;
    arg_tm->work_size[1] = 1;
    arg_tm->group_work_size[0] = 1;
    arg_tm->group_work_size[1] = 1;
    return err;
}

cl_int ocl_setup(ocl_args *a_host, ocl_args *a_dev, ocl_args *a_trans, cpx *data_in, transform_direction dir, const int group_size, const int tile_dim, const int block_dim, const int n)
{
    a_dev->n = a_host->n = n;
    a_dev->dir = a_host->dir = dir;
    ocl_setup_kernels(a_dev, group_size, true);
    memcpy(a_host, a_dev, sizeof(ocl_args));
    memcpy(a_trans, a_dev, sizeof(ocl_args));

    ocl_check_err(ocl_setup_program("ocl_kernel", "ocl_kernel_local_row", a_dev), "Failed to setup GPU Program!");
    ocl_check_err(ocl_setup_program("ocl_kernel", "ocl_kernel_global_row", a_host), "Failed to setup CPU Program!");
    ocl_check_err(ocl_setup_program("ocl_kernel", "ocl_transpose_kernel", a_trans, tile_dim, block_dim), "Failed to setup Transpose Program!");

    cl_int err = oclSetupDeviceMemoryData(a_dev, data_in);
    memcpy(a_host->work_size, a_dev->work_size, sizeof(size_t) * 3);
    memcpy(a_host->group_work_size, a_dev->group_work_size, sizeof(size_t) * 3);
    setWorkDimForTranspose(a_trans, tile_dim, block_dim, n);
    a_trans->input = a_host->input = a_dev->input;
    a_trans->output = a_host->output = a_dev->output;
    a_trans->data_mem_size = a_host->data_mem_size = a_dev->data_mem_size;
    a_host->workDim = 3;
    a_dev->workDim = 3;
    a_trans->workDim = 3;
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
    if (dref != NULL)   res = diff_seq(*din, *dref, batch_size(n)) > RELATIVE_ERROR_MARGIN;
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
    size_t minSize = n * n * sizeof(cpx);
    cpx *_in = (cpx *)malloc(minSize);
    if (_in) {
        if (out != NULL)
            *out = (cpx *)malloc(minSize);
        read_image(_in, input_file, &sz);
        *ref = (cpx *)malloc(minSize);
        memcpy(*ref, _in, minSize);
        *in = _in;
    }
}

double ocl_get_elapsed(cl_event s, cl_event e)
{
    cl_ulong start = 0, end = 0;
    clWaitForEvents(1, &e);
    clGetEventProfilingInfo(s, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    return (double)(end - start)*(cl_double)(1e-03);
}

int ocl_platformname_to_id(std::string s1)
{
    if (s1.find("NVIDIA") != std::string::npos)
        return VENDOR_NVIDIA;
    else if (s1.find("AMD") != std::string::npos)
        return VENDOR_AMD;
    return -1;
}

cl_int ocl_get_platform(cl_platform_id *platform_id)
{
    cl_uint number_of_platforms;
    cl_platform_id platforms[3];
    ocl_check_err(clGetPlatformIDs(3, platforms, &number_of_platforms), "clGetPlatformIDs");
    for (cl_platform_id pid : platforms) {
        char name[256];
        clGetPlatformInfo(pid, CL_PLATFORM_NAME, 256, name, NULL);
        if (vendor_gpu == ocl_platformname_to_id(name)) {
            *platform_id = pid;
            return CL_SUCCESS;
        }
    }
    return CL_INVALID_PLATFORM;
}