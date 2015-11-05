#version 430
#define WIDTH 8192
#define TILE_DIM 64

// Tweak here!
#define BLOCK_DIM 32

struct cpx
{
    float x;
    float y;
};

layout(local_size_x = BLOCK_DIM, local_size_y = BLOCK_DIM) in;

layout(std430, binding = 0) buffer ssbo0{ cpx input_data[]; };
layout(std430, binding = 1) buffer ssbo1{ cpx output_data[]; };

shared cpx tile[TILE_DIM][TILE_DIM + 1];

void main()
{
    uint i, j;
    uint x = (gl_WorkGroupID.x * TILE_DIM) + gl_LocalInvocationID.x;
    uint y = (gl_WorkGroupID.y * TILE_DIM) + gl_LocalInvocationID.y;
    for (j = 0; j < TILE_DIM; j += BLOCK_DIM) {
        for (i = 0; i < TILE_DIM; i += BLOCK_DIM) {
            tile[gl_LocalInvocationID.y + j][gl_LocalInvocationID.x + i] = input_data[((y + j) * WIDTH) + (x + i)];
        }
    }

    barrier();

    x = gl_WorkGroupID.y * TILE_DIM + gl_LocalInvocationID.x;
    y = gl_WorkGroupID.x * TILE_DIM + gl_LocalInvocationID.y;
    for (j = 0; j < TILE_DIM; j += BLOCK_DIM) {
        for (i = 0; i < TILE_DIM; i += BLOCK_DIM) {
            output_data[((y + j) * WIDTH) + (x + i)] = tile[gl_LocalInvocationID.x + i][gl_LocalInvocationID.y + j];
        }
    }
}