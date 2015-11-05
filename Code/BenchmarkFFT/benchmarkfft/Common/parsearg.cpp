#include "parsearg.h"

#define MATCH(s) (str.compare((s)) == 0)
#define MATCHP(s) (tmp.compare((s)) == 0)

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
            arg->start = atoi(argv[++i]);
            arg->end = atoi(argv[++i]);
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
            arg->number_of_lengths = arg->end - arg->start + 1;
        }
        else if (MATCH("-platforms")) {
            ++i;
            while (i < argc && std::string(argv[i])[0] != '-') {
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
                else if (MATCHP("id3dx11")) arg->platform_id3dx11 = true;   // DirectX lib
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
        else if (MATCH("-cuprop")) {
            arg->show_cuprop = true;
        }
        else if (MATCH("-testground")) {
            arg->run_testground = true;
        }
        else {
            printf("Unknown argument: %s\n", argv[i]);
            goto show_usage;
        }
    }
    if (arg->dimensions == 2) {
        if (arg->start < log2_32(64)) {
            arg->start = log2_32(64);
            arg->number_of_lengths = arg->end - arg->start + 1;
            printf("Notice: start exponent is set to %d\n", arg->start);
        }
        if (arg->end > HIGHEST_EXP_2D) {
            arg->end = HIGHEST_EXP_2D;
            arg->number_of_lengths = arg->end - arg->start + 1;
            printf("Notice: end exponent is set to %d\n", arg->end);
        }
    }
    return 1;
show_usage:
    printf("usage: %s [-dim #dimensions] [-r start_exponent last_exponent] [-platforms p1...] [-v] [-t] [-d] [-img] [-p] [-profiler] [-cuprop]", argv[0]);
    return 0;
}