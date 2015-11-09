#include "MyOpenCL.h"

MyOpenCL::MyOpenCL(const int dim, const int runs)
    : Platform(dim)
{
    name = "OpenCL";

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
    printf("OpenCL Platform: %s\nOpenCL Device: %s\n", info_platform, info_device);
}

MyOpenCL::~MyOpenCL()
{
}

bool MyOpenCL::validate(const int n, bool write_img)
{   
    if (dimensions == 1)
        return ocl_validate(n) == 1;
    return ocl_2d_validate(n, write_img) == 1;
}

void MyOpenCL::runPerformance(const int n)
{
    double time = ((dimensions == 1) ? ocl_performance(n) : ocl_2d_performance(n));
    results.push_back(time);
}
