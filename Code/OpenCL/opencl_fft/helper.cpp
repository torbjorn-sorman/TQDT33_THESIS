
#include "helper.h"

#define ERROR_MARGIN 0.0001

static LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds, Frequency;

void startTimer()
{
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);
}

double stopTimer()
{
    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    return (double)ElapsedMicroseconds.QuadPart;
}

std::string getKernel(const char *filename)
{
    std::ifstream in(filename);
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());    
    return contents;
}

unsigned int power(const unsigned int base, const unsigned int exp)
{
    if (exp == 0)
        return 1;
    unsigned int value = base;
    for (unsigned int i = 1; i < exp; ++i) {
        value *= base;
    }
    return value;
}

unsigned int power2(const unsigned int exp)
{
    return power(2, exp);
}

int checkError(cpx *seq, cpx *ref, float refScale, const int n, int print)
{
    int j;
    double re, im, i_val, r_val;
    re = im = 0.0;
    for (j = 0; j < n; ++j) {
        r_val = abs(refScale * seq[j].x - ref[j].x);
        i_val = abs(refScale * seq[j].y - ref[j].y);
        re = re > r_val ? re : r_val;
        im = im > i_val ? im : i_val;
    }
    if (print == 1) printf("Error\tre(e): %f\t im(e): %f\t@%u\n", re, im, n);
    return re > ERROR_MARGIN || im > ERROR_MARGIN;
}

int checkError(cpx *seq, cpx *ref, const int n, int print)
{
    return checkError(seq, ref, 1.f, n, print);
}

int checkError(cpx *seq, cpx *ref, const int n)
{
    return checkError(seq, ref, n, 0);
}

cpx *get_seq(const int n)
{
    return get_seq(n, 0);
}

cpx *get_seq(const int n, const int sinus)
{
    int i;
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].x = sinus == 0 ? 0.f : (float)sin(M_2_PI * (((double)i) / n));
        seq[i].y = 0.f;
    }
    return seq;
}

cpx *get_seq(const int n, cpx *src)
{
    int i;
    cpx *seq;
    seq = (cpx *)malloc(sizeof(cpx) * n);
    for (i = 0; i < n; ++i) {
        seq[i].x = src[i].x;
        seq[i].y = src[i].y;
    }
    return seq;
}

void write_console(float x)
{
    if (x == 0)
        printf(" %.3f", 0.f);
    else {
        if (x > 0)
            printf(" ");
        printf("%.3f", x);
    }
}

void write_console(cpx a)
{
    write_console(a.x);
    printf("\t");
    write_console(a.y);
}

void write_console(cpx *seq, const int n)
{
    for (int i = 0; i < n; ++i){
        write_console(seq[i]);
        printf("\n");
    }
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
        printf("Error: %d %s\n", args, msg);
        return 1;
    }
    return 0;
}

typedef struct {
    size_t global_work_size[3] = { 1, 1, 1 };
    size_t local_work_size[3] = { 1, 1, 1 };
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    cl_mem input, output;
    cl_platform_id platform;
} oclParams;

cl_int oclSetup(char *kernelName, cpx *dev_in, const int n, oclParams *params)
{
    oclParams p;
    size_t global_work_size[3] = { 1, 1, 1 };
    size_t local_work_size[3] = { 1, 1, 1 };

    cl_int err = CL_SUCCESS;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_mem input, output;
    cl_platform_id platform;

    unsigned int no_plat;
    err = clGetPlatformIDs(1, &platform, &no_plat);
    if (checkErr(err, no_plat, "Failed to get platform!")) return err;

    // Connect to a compute device    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (checkErr(err, "Failed to create a device group!")) return err;

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (checkErr(err, "Failed to create a compute context!")) return err;

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (checkErr(err, "Failed to create a command commands!")) return err;

    std::string filename = kernelName;
    filename += ".cl";
    std::string data = getKernel(filename.c_str());

    char *kernelSrc = (char *)malloc(sizeof(char) * (data.size() + 1));
    strcpy_s(kernelSrc, data.size() + 1, data.c_str());
    kernelSrc[data.size()] = '\0';

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSrc, NULL, &err);
    if (checkErr(err, "Failed to create compute program!")) return err;

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        printf("Error: Failed to build program executable! Len: %d\n", len);
        char *buffer = (char *)malloc(sizeof(char) * len);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("%s\n", buffer);
        free(buffer);
        return err;
    }

    // Create the compute kernel in the program we wish to run    
    kernel = clCreateKernel(program, kernelName, &err);
    if (checkErr(err, "Failed to create compute kernel!") || !kernel) return err;

    const int n2 = n / 2;
    global_work_size[0] = n2;
    local_work_size[0] = (n2 > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : n2);

    const int nBlock = n / global_work_size[0];
    size_t shMemSize = sizeof(cpx) * local_work_size[0] * 2;

    // Create the input and output arrays in device memory for our calculation
    input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cpx) * n, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cpx) * n, NULL, NULL);
    if (!input || !output) {
        printf("Error: Failed to allocate device memory!\n");
        return err;
    }

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(cpx) * n, dev_in, 0, NULL, NULL);
    if (checkErr(err, "Failed to write to source array!")) return err;
    
}

cl_int oclSetArgs()
{
    const float w_angle = dir * (M_2_PI / n);
    const float w_bangle = dir * (M_2_PI / nBlock);
    const cpx scale = { (dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f };
    const int depth = log2_32(n);
    const int breakSize = log2_32(MAX_BLOCK_SIZE);

    // Set the arguments to our compute kernel
    cl_int err = 0;
    int arg = 0;
    err = clSetKernelArg(kernel, arg++, shMemSize, NULL);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &w_angle);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &w_bangle);
    err |= clSetKernelArg(kernel, arg++, sizeof(int), &depth);
    err |= clSetKernelArg(kernel, arg++, sizeof(int), &breakSize);
    err |= clSetKernelArg(kernel, arg++, sizeof(cpx), &scale);
    err |= clSetKernelArg(kernel, arg++, sizeof(int), &nBlock);
    err |= clSetKernelArg(kernel, arg++, sizeof(int), &n2);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return err;
    }
}

cl_int oclExecute()
{
    cl_int err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    clFinish(commands);
    return err;
}

void oclRelease(cpx *dev_out, const int n)
{
    
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(cpx) * n, dev_out, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        return;
    }

    // Shutdown and cleanup
    free(kernelSrc);
    //clReleaseMemObject(shared);
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

}

