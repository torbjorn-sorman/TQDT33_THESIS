#include "tsCombine2.cuh"

typedef int syncVal;

__global__ void _kernelAll2(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist);
__global__ void _kernelGPUS2(cpx *in, cpx *out, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, int n2);
__global__ void _kernelBlock2(cpx *in, cpx *out, float angle, const cpx scale, int depth, unsigned int lead, int n2);

__device__ static __inline__ void algorithm_p2(cpx *shared, int in_high, float angle, int bit);
__device__ static __inline__ void inner_k2(cpx *in, cpx *out, float angle, int steps, unsigned int lmask, int dist);

__host__ int tsCombine2_Validate(int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombine2(FFT_FORWARD, &dev_in, &dev_out, n);
    tsCombine2(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsCombine2_Performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombine2(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

// My device specifics!
// Seven physical "cores" that can run blocks in "parallel" (and most important sync threads in a block). 1024 is the thread limit per physical core.
// Essentially (depending on scheduling and other factors) # blocks fewer than HW_LIMIT can be synched, any # above is not trivially solved. cuFFT solves this.
#define HW_LIMIT (1024 / MAX_BLOCK_SIZE) * 7

__host__ void tsCombine2(fftDir dir, cpx **dev_in, cpx **dev_out, int n)
{
    int threads, blocks;
    int depth = log2_32(n);
    const int lead = 32 - depth;
    const int n2 = (n / 2);
    const int breakSize = log2_32(MAX_BLOCK_SIZE);
    cpx scaleCpx = make_cuFloatComplex((dir == FFT_FORWARD ? 1.f : 1.f / n), 0.f);
    set_block_and_threads(&blocks, &threads, n2);
    int numBlocks = blocks;
    const int nBlock = n / blocks;    
    const float w_angle = dir * (M_2_PI / n);
    const float w_bangle = dir * (M_2_PI / nBlock);
    int bSize = n2;   
     
    if (blocks >= HW_LIMIT) {
        // Calculate sequence until parts fit into a block, syncronize on CPU until then.
        --depth;
        int steps = 0;        
        int dist = n2;
        _kernelAll2 KERNEL_ARGS2(blocks, threads)(*dev_in, *dev_out, w_angle, 0xFFFFFFFF << depth, steps, dist);
        cudaDeviceSynchronize();
        while (--depth > breakSize) {
            dist >>= 1;
            ++steps;
            _kernelAll2 KERNEL_ARGS2(blocks, threads)(*dev_out, *dev_out, w_angle, 0xFFFFFFFF << depth, steps, dist);
            cudaDeviceSynchronize();
        }
        swap(dev_in, dev_out);
        ++depth;
        bSize = nBlock / 2;
        numBlocks = 1;
    }

    // Calculate complete sequence in one launch and syncronize on GPU
    _kernelGPUS2 KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, w_angle, w_bangle, depth, lead, breakSize, scaleCpx, numBlocks, bSize);
    cudaDeviceSynchronize();

}

__device__ static __inline__ void butterfly(cpx in_low, cpx in_high, cpx *out, int low, int high, float angle)
{
    cpx w;
    SIN_COS_F(angle, &w.y, &w.x);
    cpx_add_sub_mul(&(out[low]), &(out[high]), in_low, in_high, w);
}

// Take no usage of shared mem yet...
__global__ void _kernelAll2(cpx *in, cpx *out, float angle, unsigned int lmask, int steps, int dist)
{
    inner_k2(in, out, angle, steps, lmask, dist);
}

__device__ static __inline__ void inner_k2(cpx *in, cpx *out, float angle, int steps, unsigned int lmask, int dist)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int in_low = tid + (tid & lmask);
    int in_high = in_low + dist;
    butterfly(in[in_low], in[in_high], out, in_low, in_high, angle * ((tid << steps) & ((dist - 1) << steps)));
}

__device__ static __inline__ int algorithm_c2(cpx *in, cpx *out, int bit_start, int breakSize, float angle, int nBlocks, int dist)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int steps = 0;
    init_sync(tid, nBlocks);
    inner_k2(in, out, angle, steps, 0xFFFFFFFF << bit_start, dist);
    __gpu_sync(nBlocks + steps);
    for (int bit = bit_start - 1; bit > breakSize; --bit) {
        dist = dist >> 1;
        ++steps;
        inner_k2(out, out, angle, steps, 0xFFFFFFFF << bit, dist);
        __gpu_sync(nBlocks + steps);
    }
    return breakSize + 1;
}

__device__ static __inline__ void algorithm_p2(cpx *shared, int in_high, float angle, int bit)
{
    cpx in_lower, in_upper;
    int i = (threadIdx.x << 1);
    int ii = i + 1;
    for (int steps = 0; steps < bit; ++steps) {
        in_lower = shared[threadIdx.x];
        in_upper = shared[in_high];
        SYNC_THREADS;
        butterfly(in_lower, in_upper, shared, i, ii, angle * ((threadIdx.x & (0xFFFFFFFF << steps))));
        SYNC_THREADS;
    }
}

// Full blown block syncronized algorithm! In theory this should scale up but is limited by hardware (#cores)
__global__ void _kernelGPUS2(cpx *in, cpx *out, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, int n2)
{
    extern __shared__ cpx shared[];
    int bit = depth;
    int in_high = n2;
    if (nBlocks > 1) {
        bit = algorithm_c2(in, out, depth - 1, breakSize, angle, gridDim.x, in_high);
        in_high >>= log2(nBlocks);
        in = out;
    }
    int offset = blockIdx.x * blockDim.x * 2;
    in_high += threadIdx.x;
    mem_gtos(threadIdx.x, in_high, offset, shared, in);
    algorithm_p2(shared, in_high, bAngle, bit);
    mem_stog_db(threadIdx.x, in_high, offset, lead, scale, shared, out);
}