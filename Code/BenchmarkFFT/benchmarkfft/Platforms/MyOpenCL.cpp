#include "MyOpenCL.h"

MyOpenCL::MyOpenCL(const int dim, const int runs)
    : Platform(dim)
{
    name = "OpenCL";

    cl_platform_id platform;
    char info[255];
    size_t actual;

    clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 255, info, &actual);
    printf("OpenCL version: %s\n", info);
    if (actual > 255) {
        printf("Do stuff...");
    }
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
