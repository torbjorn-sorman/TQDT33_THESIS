#include "MyClFFT.h"

#include <clFFT.h>
#include <clAmdFFT.h>
#include "../Definitions.h"
#include "OpenCL/ocl_helper.h"
#include "../Common/mathutil.h"
#include "../Common/mytimer.h"

MyClFFT::MyClFFT(const int dim, const int runs)
    : Platform(dim)
{
    name = "clFFT";

    cl_platform_id platform;
    cl_device_id device;
    char info_platform[255];
    char info_device[255];
    size_t actual;

    clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 255, info_platform, &actual);
    if (actual > 255) {
        printf("Do stuff...");
    }
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 255, info_device, &actual);    
    if (actual > 255) {
        printf("Do stuff...");
    }
    printf("ClFFT Platform: %s\nClFFT Device: %s\n", info_platform, info_device);
}

MyClFFT::~MyClFFT()
{
}

bool MyClFFT::validate(const int n, bool write_img)
{   
    return true;
}

cl_int setTimeKernel(cl_context context, cl_device_id device_id, ocl_args *args)
{
    cl_int err = CL_SUCCESS;
    cl_program program;
    char *src = get_kernel_src(std::string("Platforms/OpenCL/ocl_kernel.cl"), NULL);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&src, NULL, &err);
    if (err != CL_SUCCESS)
        return err;

    // Build the program executable
    if (err = clBuildProgram(program, 0, NULL, "-cl-single-precision-constant -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
        char *buffer = (char *)malloc(sizeof(char) * len);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("\n%s\n", buffer);
        free(buffer);
        return err;
    }

    // Create the compute kernel in the program we wish to run    
    args->kernel = clCreateKernel(program, "ocl_timestamp_kernel", &err);
    if (err != CL_SUCCESS)
        return err;

    args->program = program;
    free(src);
    return err;
}

void MyClFFT::runPerformance(const int n)
{
    bool dim1 = dimensions == 1;
    double measurements[64];
    double time = 0.0;

    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufX;
    cl_event event = NULL;
    clfftPlanHandle planHandle;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    size_t ret_param_size = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    ctx = clCreateContext(0, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    
    size_t clLengths1D[1] = { n };
    size_t clLengths2D[2] = { n, n };

    /* Setup clFFT. */
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);

    /* Create a default plan for a complex FFT. */
    err = clfftCreateDefaultPlan(&planHandle, ctx, dim1 ? CLFFT_1D : CLFFT_2D, dim1 ? clLengths1D : clLengths2D);

    /* Set plan parameters. */
    err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
    err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
    /* Bake the plan. */
    err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
    /* Prepare OpenCL memory objects and place data inside them. */
    bufX = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (dim1 ? n : n * n) * sizeof(cpx), NULL, &err);

        for (int i = 0; i < number_of_tests; ++i) {                    
            start_timer();
            clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, NULL, NULL);
            clFinish(queue);
            measurements[i] = stop_timer();
        }
        time = average_best(measurements, number_of_tests);

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufX);
    /* Release the plan. */
    err = clfftDestroyPlan(&planHandle);
    /* Release clFFT library. */
    clfftTeardown();
    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    results.push_back(time);
}
