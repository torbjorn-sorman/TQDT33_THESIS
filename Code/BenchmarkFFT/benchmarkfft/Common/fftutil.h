#pragma once
#ifndef FFTUTIL_H
#define FFTUTIL_H

#include "../Definitions.h"
#include "mathutil.h"

struct fft_args
{
    float global_angle;
    float local_angle;
    float scalar;
    int leading_bits;
    int steps; 
    int steps_left;
    int steps_gpu;
    int dist;
    int block_range;
    int n_per_block;
};

static __inline void set_fft_arguments(fft_args *args, transform_direction dir, const int blocks, const int block_size, const int n)
{
    args->n_per_block = n / blocks;
    args->local_angle = dir * (M_2_PI / args->n_per_block);
    args->steps_left = log2_32(n);
    args->leading_bits = 32 - args->steps_left;
    args->scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
    if (blocks > 1) {
        args->global_angle = dir * (M_2_PI / n);
        args->steps = 0;
        args->dist = n;
        args->steps_gpu = log2_32(block_size);
        args->block_range = args->n_per_block >> 1;
    }
    else {
        args->block_range = n >> 1;
    }
}

static __inline void set_fft_arguments(fft_args *args, const int blocks, const int n)
{
    args->steps_left = log2_32(n);
    if (blocks > 1) {
        args->steps = 0;
        args->dist = n;
    } 
}

#endif