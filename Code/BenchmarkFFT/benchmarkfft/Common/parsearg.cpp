#include "parsearg.h"

#define MATCH(s) (str.compare((s)) == 0)
#define MATCHP(s) (tmp.compare((s)) == 0)

int count_sub_arguments(char *argv[], int index, int argc)
{
    int i;
    for (i = 0; i < argc - index; ++i) {
        if (argv[index + i][0] == '-')
            return i;

    }
    return i;
}

bool valid_content(int i, int argc, char* argv[])
{
    return i < argc && std::string(argv[i])[0] != '-';
}

int parse_args(benchmarkArgument *arg, int argc, char* argv[])
{
    if (argc == 1) {
        goto show_usage;
    }
    for (int i = 1; i < argc; ++i) {        
        std::string str = argv[i];
        if (MATCH("-dim")) {
            arg->dimensions = atoi(argv[++i]);
            if (arg->dimensions < 1 || arg->dimensions > 2) {
                printf("Error parsing #dimensions: must be 1 or 2\n");
                goto show_usage;
            }
        }
        else if (MATCH("-r")) {
            int range_args = count_sub_arguments(argv, i + 1, argc);
            if (range_args == 1) {
                arg->start = arg->end = atoi(argv[++i]);
            }
            else if (range_args == 2) {
                arg->start = atoi(argv[++i]);
                arg->end = atoi(argv[++i]);
            }
            else {
                goto show_usage;
            }
            if (arg->end > HIGHEST_EXP) {
                arg->end = HIGHEST_EXP;
                printf("Notice: end exponent is set to %d\n", arg->start);
            }
            if (arg->start > HIGHEST_EXP) {
                arg->start = HIGHEST_EXP;
                printf("Notice: start exponent is set to %d\n", arg->start);
            }
            if (arg->end < arg->start) {
                printf("Error parsing range of lengths, start exponent must be lower than end exponent.\n");
                goto show_usage;
            }
            arg->test_runs = arg->end - arg->start + 1;
        }
        else if (MATCH("-platforms")) {
            ++i;
            //while (i < argc && std::string(argv[i])[0] != '-') {
            while (valid_content(i, argc, argv)) {
                arg->test_platform = true;
                std::string tmp = argv[i];
                if      (MATCHP("c"))       arg->platform_c = true;
                else if (MATCHP("cu"))      arg->platform_cuda = true;
                else if (MATCHP("dx"))      arg->platform_directx = true;
                else if (MATCHP("ocl"))     arg->platform_opencl = true;
                else if (MATCHP("gl"))      arg->platform_opengl = true;
                else if (MATCHP("omp"))     arg->platform_openmp = true;
                else if (MATCHP("fftw"))    arg->platform_fftw = true;      // Open Src lib
                else if (MATCHP("cufft"))   arg->platform_cufft = true;     // NVidia lib
                else if (MATCHP("clfft"))   arg->platform_clfft = true;     // AMD OpenCL lib
                else if (MATCHP("dx11"))    arg->platform_id3dx11 = true;   // DirectX lib
                else {
                    printf("Unknown platform: %s\n", tmp.c_str());
                    goto show_usage;
                }
                ++i;
            }
            --i;
        }
        else if (MATCH("-v")) {
            arg->validate = true;
        }
        else if (MATCH("-t")) {
            arg->performance_metrics = true;
        }
        else if (MATCH("-d")) {
            arg->display = true;
        }
        else if (MATCH("-img")) {
            arg->write_img = true;
        }
        else if (MATCH("-profiler")) {
            arg->profiler = true;            
        }
        else if (MATCH("-p")) {
            arg->write_file = true;
        }
        else if (MATCH("-show_properties")) {
            arg->show_device_properties = true;
        }
        else if (MATCH("-testground")) {
            arg->run_testground = true;
        }
        else if (0 && MATCH("-vendor")) {
            ++i;
            if (valid_content(i, argc, argv) && count_sub_arguments(argv, i, argc) == 1) {
                std::string tmp = argv[i];
                if (MATCHP("nvidia"))       arg->vendor = VENDOR_NVIDIA;
                else if (MATCHP("amd"))     arg->vendor = VENDOR_AMD;
                else if (MATCHP("basic"))   arg->vendor = VENDOR_BASIC;
                else {
                    printf("Unknown vendor: %s, must be one of {nvidia, amd}\n", tmp.c_str());
                    goto show_usage;
                }
            }
            else {
                goto show_usage;
            }
        }
        else if (MATCH("-n_tests")) {
            int n_tests = atoi(argv[++i]);
            if (n_tests > 0 && n_tests <= 64)
                number_of_tests = n_tests;
            else
                printf("Number of runs is limited to [1 : 64], default is %d", number_of_tests);
        }
        else {
            printf("Unknown argument: %s\n", argv[i]);
            goto show_usage;
        }
    }
    if (arg->dimensions == 2) {
        if (arg->start < log2_32(64)) {
            arg->start = log2_32(64);
            arg->test_runs = arg->end - arg->start + 1;
            printf("Notice: start exponent is set to %d\n", arg->start);
        }
        if (arg->end > HIGHEST_EXP_2D) {
            arg->end = HIGHEST_EXP_2D;
            arg->test_runs = arg->end - arg->start + 1;
            printf("Notice: end exponent is set to %d\n", arg->end);
        }
    }
    return 1;
show_usage:
    printf("usage: %s [-dim #dimensions] [-r start_exponent last_exponent] [-platforms p1...] [-v] [-t] [-d] [-img] [-p] [-profiler] [-cuprop]", argv[0]);
    return 0;
}