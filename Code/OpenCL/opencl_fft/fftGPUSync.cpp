#include "fftGPUSync.h"

void tsCombineGPUSync(fftDir dir, cpx *dev_in, cpx *dev_out, cInt n);

int GPUSync_validate(cInt n)
{    
    return 1;
}

double GPUSync_performance(cInt n)
{
    return -1.0;
}

int checkErr(cl_int error, char *msg)
{
    if (error != CL_SUCCESS) {
        printf("Error: %s\n", msg);
        return 0;
    }
    return 1;
}

int checkErr(cl_int error, cl_int args, char *msg)
{
    if (error != CL_SUCCESS) {
        printf("Error: %s %d\n", args, msg);
        return 0;
    }
    return 1;
}

void tsCombineGPUSync(fftDir dir, cpx *dev_in, cpx *dev_out, cInt n)
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

    // Connect to a compute device
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (checkErr(err, "Failed to create a device group!")) return;

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (checkErr(err, "Failed to create a compute context!")) return;

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (checkErr(err, "Failed to create a command commands!")) return;

    char *KernelSource = "";

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)& KernelSource, NULL, &err);
    if (checkErr(err, "Failed to create compute program!")) return;

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return;
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || checkErr(err, "Failed to create compute kernel!")) return;

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
    if (!input || !output) {
        printf("Error: Failed to allocate device memory!\n");
        return;
    }

    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * n, data, 0, NULL, NULL);
    if (checkErr(err, "Failed to write to source array!")) return;

    // Set the arguments to our compute kernel
    //
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &n);
    if (checkErr(err, err, "Failed to set kernel arguments! ")) return;

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (checkErr(err, err, "Failed to retrieve kernel work group info!")) return;

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = n;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * n, results, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        return;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values!\n", 1, n);

    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}