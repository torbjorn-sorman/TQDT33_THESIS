
struct cpx
{
    float x;
    float y;
};

// MAX_BLOCK_SIZE specifies the number of threads in a group; this
// must be the same number as the constant value groupSize defined
// in the CPU code.
#define MAX_BLOCK_SIZE 1024

cbuffer dx_cs_args
{
    float angle;
    float local_angle;
    int steps_left;
    int leading_bits;
    int steps_gpu;
    float scalar;
    int number_of_blocks;
    int n_half;
};

StructuredBuffer<cpx> dev_in;
StructuredBuffer<cpx> dev_out;

[numthreads(MAX_BLOCK_SIZE, 1, 1)]
void dx_fft(uint3 threadIDInGroup : SV_GroupThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    
    // Compute the index of the element to be processed by this thread.
    //int n = groupID.x * MAX_BLOCK_SIZE + threadIDInGroup.x;

    // Compute the output value z from input buffers x, y and the constant
    // value a (which is defined up above in the "Constants" buffer).
    //z[n] = a * x[n] + y[n];
}

/*

int cuda_algorithm_global_sync(cpx *in, cpx *out, int bit_start, int steps_gpu, float angle, int number_of_blocks, int n_half)
{
int dist = n_half;
int steps = 0;
cuda_block_sync_init(sync_array_in, sync_array_out, (blockIdx.x * blockDim.x + threadIdx.x), number_of_blocks);
cuda_inner_kernel(in, out, angle, steps, 0xFFFFFFFF << bit_start, dist);
cuda_block_sync(sync_array_in, sync_array_out, number_of_blocks + steps);
for (int bit = bit_start - 1; bit > steps_gpu; --bit) {
dist >>= 1;
++steps;
cuda_inner_kernel(out, out, angle, steps, 0xFFFFFFFF << bit, dist);
cuda_block_sync(sync_array_in, sync_array_out, number_of_blocks + steps);
}
return steps_gpu + 1;
}

void cuda_algorithm_local(cpx *shared, int in_high, float angle, int bit)
{
cpx w, in_lower, in_upper;

cpx *out_i = shared + (threadIdx.x << 1);
cpx *out_ii = out_i + 1;
cpx *in_l = shared + threadIdx.x;
cpx *in_u = shared + in_high;
for (int steps = 0; steps < bit; ++steps) {
SIN_COS_F(angle * (threadIdx.x & (0xFFFFFFFF << steps)), &w.y, &w.x);
in_lower = *in_l;
in_upper = *in_u;
SYNC_THREADS;
cpx_add_sub_mul(&in_lower, &in_upper, out_i, out_ii, &w);
SYNC_THREADS;
}
}

// Full blown block syncronized algorithm! In theory this should scalar up but is limited by hardware (#cores)
void cuda_kernel_local(cpx *in, cpx *out, float angle, float local_angle, int steps_left, int leading_bits, int steps_gpu, float scalar, int number_of_blocks, int n_half)
{
extern __shared__ cpx shared[];
int bit = steps_left;
int in_high = n_half;
if (number_of_blocks > 1) {
bit = cuda_algorithm_global_sync(in, out, steps_left - 1, steps_gpu, angle, number_of_blocks, in_high);
in_high >>= log2(number_of_blocks);
in = out;
}

int offset = blockIdx.x * blockDim.x * 2;
in_high += threadIdx.x;
in += offset;
shared[threadIdx.x] = in[threadIdx.x];
shared[in_high] = in[in_high];
SYNC_THREADS;
cuda_algorithm_local(shared, in_high, local_angle, bit);


out[BIT_REVERSE(threadIdx.x + offset, leading_bits)] = { shared[threadIdx.x].x * scalar, shared[threadIdx.x].y *scalar };
out[BIT_REVERSE(in_high + offset, leading_bits)] = { shared[in_high].x * scalar, shared[in_high].y *scalar };
}
*/