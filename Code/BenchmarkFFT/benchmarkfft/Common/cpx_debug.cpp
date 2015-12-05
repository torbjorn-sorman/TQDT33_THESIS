#include "cpx_debug.h"
#include <iostream>
#include <sstream>

void cpx_to_console(cpx *sequence, char *title, int len)
{
    if (sequence == NULL) {
        printf("\tSequence is NULL\n");
        return;
    }
    std::cout << title << ":" << std::endl;
    for (int i = 0; i < len; ++i)
        printf("\t%d: %f\t%f\n", i, sequence[i].x, sequence[i].y);
}

void cpx2d_to_console(cpx *sequence, char *title, int len)
{
    std::cout << title << ":" << std::endl;
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 7; ++x) {
            cpx *c = sequence + y * len + x;
            printf("(%.0f_%.0f)", c->x, c->y);
        }
        std::cout << std::endl;
    }
}

void debug_check_compare(const int n)
{
    if (debug_cuda_out != NULL && debug_dx_out != NULL) {
        printf("Debug Diff: %f\n", diff_seq(debug_dx_out, debug_cuda_out, n));
        free_all(debug_cuda_out, debug_dx_out);
    }
}

/*struct ocl_args {
    int n;
    int n_per_block;
    int number_of_blocks;
    float dir;
    cl_uint workDim;
    size_t data_mem_size;
    size_t work_size[3];
    size_t group_work_size[3];
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    cl_mem input, output;
    cl_platform_id platform;
};*/

void debug_ocl_args(ocl_args *arg)
{
    std::stringstream str;
    str << std::endl;
    str << "n:\t\t" << arg->n << std::endl;
    //str << "n_per_block:\t" << arg->n_per_block << std::endl;
    str << "number_of_blks:\t" << arg->number_of_blocks << std::endl;
    str << "dir:\t\t" << arg->dir << std::endl;
    str << "workDim:\t" << arg->workDim << std::endl;
    str << "data_mem_size:\t" << arg->data_mem_size << std::endl;
    str << "work_size:\t" << arg->work_size[0] << "\t" << arg->work_size[1] << "\t" << arg->work_size[2] << std::endl;
    str << "group_work_sz:\t" << arg->group_work_size[0] << "\t" << arg->group_work_size[1] << "\t" << arg->group_work_size[2] << std::endl;
    std::cout << str.str();    
}