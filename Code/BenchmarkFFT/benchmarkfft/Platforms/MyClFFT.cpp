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

    cl_platform_id platform_id;
    ocl_check_err(ocl_get_platform(&platform_id), "ocl_get_platform");
    cl_device_id device;
    char info_platform[511];
    char info_device[511];
    size_t actual;

    clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 511, info_platform, &actual);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 511, info_device, &actual);
    printf("OpenCL:\t\t%s (platform)\n\t\t%s (device)\n", info_platform, info_device);
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
    cl_event event = NULL, start_event, end_event;
    clfftPlanHandle planHandle;

    /* Setup OpenCL environment. */
    ocl_check_err(ocl_get_platform(&platform), "ocl_get_platform");
    size_t ret_param_size = 0;
    //err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    ocl_check_err(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL), "clGetDeviceIDs");
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
    err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
    /* Bake the plan. */
    err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
    /* Prepare OpenCL memory objects and place data inside them. */
    cl_mem buf_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (dim1 ? n : n * n) * sizeof(cpx), NULL, &err);
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (dim1 ? n : n * n) * sizeof(cpx), NULL, &err);
    cl_event e;


    ocl_args tm;
    ocl_check_err(setTimeKernel(ctx, device, &tm), "setTimeKernel");
    clFinish(queue);
    for (int i = 0; i < number_of_tests; ++i) {
        clFinish(queue);
        start_timer();
        //clEnqueueNDRangeKernel(queue, tm.kernel, tm.workDim, NULL, tm.work_size, tm.group_work_size, 0, NULL, &start_event);
        //clWaitForEvents(1, &start_event);
        clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, &e, &buf_in, &buf_out, NULL);
        //clWaitForEvents(1, &e);
        //clEnqueueNDRangeKernel(queue, tm.kernel, tm.workDim, NULL, tm.work_size, tm.group_work_size, 0, NULL, &end_event);
        clWaitForEvents(1, &e);        
        //clWaitForEvents(1, &end_event);
        measurements[i] = stop_timer();        
        //measurements[i] = ocl_get_elapsed(start_event, end_event);
        clFinish(queue);
    }
    time = average_best(measurements, number_of_tests);
    //double tm = ocl_get_elapsed(e, e);

    //printf("tm: %f\n", tm);

    /* Release OpenCL memory objects. */
    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    /* Release the plan. */
    err = clfftDestroyPlan(&planHandle);
    /* Release clFFT library. */
    clfftTeardown();
    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    results.push_back(time);
}