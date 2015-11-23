#include "MyOpenCL.h"

#include <iostream>

MyOpenCL::MyOpenCL(const int dim, const int runs)
    : Platform(dim)
{
    name = "OpenCL";

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
