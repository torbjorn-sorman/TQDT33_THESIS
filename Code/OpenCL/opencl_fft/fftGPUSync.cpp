#include "fftGPUSync.h"

void GPUSync(fftDir dir, cpx *dev_in, cpx *dev_out, const int n);

int GPUSync_validate(const int n)
{    
    cpx *data_in = get_seq(n, 1);
    cpx *data_out = get_seq(n);
    cpx *data_ref = get_seq(n, data_in);

    GPUSync(-1.f, data_in, data_out, n);

    int res = checkError(data_out, data_ref, n);

    printf("\n");
    write_console(data_out, n);
    printf("\n");
    write_console(data_ref, n);

    free(data_in);
    free(data_out);
    free(data_ref);
    return res;
}

double GPUSync_performance(const int n)
{
    return -1.0;
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
        printf("Error: %s %d\n", args, msg);
        return 1;
    }
    return 0;
}

void GPUSync(fftDir dir, cpx *dev_in, cpx *dev_out, const int n)
{
    size_t global;
    size_t local;
    
    cl_int err = CL_SUCCESS;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    cl_platform_id platform;
    unsigned int no_plat;
    err = clGetPlatformIDs(1, &platform, &no_plat);
    if (checkErr(err, no_plat, "Failed to get platform!")) return;

    // Connect to a compute device    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (checkErr(err, "Failed to create a device group!")) return;
    
    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (checkErr(err, "Failed to create a compute context!")) return;

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (checkErr(err, "Failed to create a command commands!")) return;
        
    std::string data = getKernel("kernel.cl");
    
    char *kernelSrc = (char *)malloc(sizeof(char) * (data.size() + 1));
    strcpy_s(kernelSrc, data.size() + 1, data.c_str());    
    kernelSrc[data.size()] = '\0';

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSrc, NULL, &err);
    if (checkErr(err, "Failed to create compute program!")) return;
    
    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        //clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);    
        return;    
    }
        
    // Create the compute kernel in the program we wish to run    
    kernel = clCreateKernel(program, "kernelGPUSync", &err);
    if (checkErr(err, "Failed to create compute kernel!") || !kernel) return;

    // Create the input and output arrays in device memory for our calculation
    input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cpx) * n, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cpx) * n, NULL, NULL);
    if (!input || !output) {
        printf("Error: Failed to allocate device memory!\n");
        return;
    }
    
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(cpx) * n, dev_in, 0, NULL, NULL);
    if (checkErr(err, "Failed to write to source array!")) return;
    
    int threads, blocks;
    const int n2 = n / 2;
    if (n2 > MAX_BLOCK_SIZE) {
        blocks = n2 / MAX_BLOCK_SIZE;
        threads = MAX_BLOCK_SIZE;
    }
    else {
        blocks = 1;
        threads = n2;
    }

    const int nBlock = n / blocks;
    const float w_angle = dir * (M_2_PI / n);
    const float w_bangle = dir * (M_2_PI / nBlock);
    const cpx scale = {(dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f};
    const int depth = log2_32(n);
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    
    // Set the arguments to our compute kernel
    // __global cpx *in, __global cpx *out, const float angle, const float bAngle, const int depth, const int breakSize, const cpx scale, const int nBlocks, const int n2)
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(float), &w_angle);
    err |= clSetKernelArg(kernel, 3, sizeof(float), &w_bangle);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &depth);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &breakSize);
    err |= clSetKernelArg(kernel, 6, sizeof(cpx), &scale);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &nBlock);
    err |= clSetKernelArg(kernel, 8, sizeof(int), &n2);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return;
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (checkErr(err, err, "Failed to retrieve kernel work group info!")) return;
    
    local = threads;
    global = blocks;

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device    
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return;
    }

    // Wait for the command commands to get serviced before reading back results
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * n, dev_out, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        return;
    }
    
    // Shutdown and cleanup
    free(kernelSrc);
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
}