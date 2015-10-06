
#include "getopt.h"

int parseArguments(benchmarkArgument *arg, int argc, const char* argv[])
{
    if (argc == 1) {
        goto show_usage;
    }
    for (int i = 3; i < argc; ++i) {
        std::string str = argv[i];
        if (str.compare("-d") == 0) {
            arg->dimensions = atoi(argv[++i]);
            if (arg->dimensions < 1 || arg->dimensions > 2) {
                printf("Error parsing #dimensions: must be 1 or 2\n");
                goto show_usage;
            }
        }
        else if (str.compare("-r") == 0) {
            arg->start = atoi(argv[++i]);
            arg->end = atoi(argv[++i]);
            if (arg->end > HIGHEST_EXP) {
                arg->end = HIGHEST_EXP;
                printf("Notice: end exponent is set to %u", arg->start);
            }
            if (arg->start > HIGHEST_EXP) {
                arg->start = HIGHEST_EXP;
                printf("Notice: start exponent is set to %u", arg->start);
            }
            if (arg->end < arg->start) {
                printf("Error parsing range of lengths, start exponent must be lower than end exponent.\n");
                goto show_usage;
            }
            arg->number_of_lengths = arg->end - arg->start;
        }
        else if (str.compare("-platforms") == 0) {
            ++i;
            while (std::string(argv[i])[0] != '-') {
                arg->test_platform = 1;
                std::string tmp = argv[i];
                if (tmp.compare("cuda") == 0) arg->platform_cuda = 1;
                if (tmp.compare("opencl") == 0) arg->platform_opencl = 1;
                if (tmp.compare("opengl") == 0) arg->platform_opengl = 1;
                if (tmp.compare("directx") == 0) arg->platform_directx = 1;
                if (tmp.compare("c") == 0) arg->platform_c = 1;
                if (tmp.compare("openmp") == 0) arg->platform_openmp = 1;
                if (tmp.compare("cufft") == 0) arg->platform_cufft = 1;
                ++i;
            }
        }
        else if (str.compare("-v") == 0) {
            arg->validate = 1;
        }
        else if (str.compare("-d") == 0) {
            arg->display = 1;
        }
        else if (str.compare("-profiler") == 0) {
            arg->profiler = 1;
        }
        else if (str.compare("-p") == 0) {
            arg->write_file = 1;
        }
        else if (str.compare("-cuprop") == 0) {
            arg->show_cuprop = 1;
        }
        else {
            printf("Unknown argument: %s\n", argv[i]);
        }
    }
    if (arg->dimensions == 2) {
        if (arg->start < log2_32(TILE_DIM >> 1)) {
            arg->start = log2_32(TILE_DIM >> 1);
            arg->number_of_lengths = arg->end - arg->start;
            printf("Notice: start exponent is set to %u", arg->start);
        }
    }
    return 1;
show_usage:
    printf("usage: %s [-d #dimensions] [-r start_exponent last_exponent] [-platforms p1...] [-profiler] [-p] [-cuprop]", argv[0]);
    return 0;
}