
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

int cmp(const void *x, const void *y)
{
    double xx = *(double*)x, yy = *(double*)y;
    if (xx < yy) return -1;
    if (xx > yy) return  1;
    return 0;
}

double avg(double m[], int n)
{
    int i, cnt, end;
    double sum;
    qsort(m, n, sizeof(double), cmp);
    sum = 0.0;
    cnt = 0;
    end = n < 5 ? n - 1 : 5;
    for (i = 0; i < end; ++i) {
        sum += m[i];
        ++cnt;
    }
    return (sum / (double)cnt);
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
        printf("Error Code: %d\nMessage: %s\n", args, msg);
        return 1;
    }
    return 0;
}

cl_int oclSetup(char *kernelName, cpx *dev_in, oclArgs *args)
{
    cl_int err = CL_SUCCESS;
    args->global_work_size[0] = args->global_work_size[1] = args->global_work_size[2] = 1; 
    args->local_work_size[0] = args->local_work_size[1] = args->local_work_size[2] = 1;

    err = clGetPlatformIDs(1, &args->platform, NULL);
    if (err != CL_SUCCESS) return err;
    
    // Connect to a compute device    
    err = clGetDeviceIDs(args->platform, CL_DEVICE_TYPE_GPU, 1, &args->device_id, NULL);
    if (err != CL_SUCCESS) return err;

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
    err = clBuildProgram(args->program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
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
    args->global_work_size[0] = n2;
    args->local_work_size[0] = (n2 > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : n2);

    args->nBlock = args->n / n2;
    args->shared_mem_size = sizeof(cpx) * args->local_work_size[0] * 2;

    // Create the input and output arrays in device memory for our calculation
    args->input = clCreateBuffer(args->context, CL_MEM_READ_WRITE, sizeof(cpx) * args->n, NULL, NULL);
    args->output = clCreateBuffer(args->context, CL_MEM_READ_WRITE, sizeof(cpx) * args->n, NULL, NULL);
    args->sync_in = clCreateBuffer(args->context, CL_MEM_READ_WRITE, sizeof(int) * MAX_BLOCK_SIZE, NULL, NULL);
    args->sync_out = clCreateBuffer(args->context, CL_MEM_READ_WRITE, sizeof(int) * MAX_BLOCK_SIZE, NULL, NULL);
    
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(args->commands, args->input, CL_TRUE, 0, sizeof(cpx) * args->n, dev_in, 0, NULL, NULL);
    if (err != CL_SUCCESS) return err;
}

void oclRelease(cpx *dev_out, oclArgs *args, cl_int *error)
{
    cl_int err = CL_SUCCESS;
    err = clEnqueueReadBuffer(args->commands, args->output, CL_TRUE, 0, sizeof(cpx) * args->n, dev_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        *error = err;
        printf("\n%d\n", err);
    }
    free(args->kernelSource);
    clReleaseMemObject(args->input);
    clReleaseMemObject(args->output);
    clReleaseMemObject(args->sync_in);
    clReleaseMemObject(args->sync_out);
    clReleaseProgram(args->program);
    clReleaseKernel(args->kernel);
    clReleaseCommandQueue(args->commands);
    clReleaseContext(args->context);
}

