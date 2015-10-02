#include "tsCombine.cuh"

typedef int syncVal;

__global__ void _kernelAll(cpx *in, cpx *out, float angle, unsigned int lmask, unsigned int pmask, int steps, int dist);
__global__ void _kernelGPUS(cpx *in, cpx *out, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, int n2);
__global__ void _kernelBlock(cpx *in, cpx *out, float angle, const cpx scale, int depth, unsigned int lead, int n2);

__device__ static __inline__ void algorithm_p(cpx *shared, int in_high, float angle, int bit);
__device__ static __inline__ void inner_k(cpx *in, cpx *out, float angle, int steps, unsigned int lmask, unsigned int pmask, int dist);

__host__ int tsCombine_Validate(int n)
{
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    tsCombine(FFT_FORWARD, &dev_in, &dev_out, n);
    tsCombine(FFT_INVERSE, &dev_out, &dev_in, n);
    cudaMemcpy(in, dev_in, n * sizeof(cpx), cudaMemcpyDeviceToHost);

    return fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out) != 1;
}

__host__ double tsCombine_Performance(int n)
{
    double measures[NUM_PERFORMANCE];
    cpx *in, *ref, *out, *dev_in, *dev_out;
    fftMalloc(n, &dev_in, &dev_out, NULL, &in, &ref, &out);

    cudaMemcpy(dev_in, in, n * sizeof(cpx), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PERFORMANCE; ++i) {
        startTimer();
        tsCombine(FFT_FORWARD, &dev_in, &dev_out, n);
        measures[i] = stopTimer();
    }

    fftResultAndFree(n, &dev_in, &dev_out, NULL, &in, &ref, &out);
    return avg(measures, NUM_PERFORMANCE);
}

// My device specifics!
// Seven physical "cores" that can run blocks in "parallel" (and most important sync threads in a block). 1024 is the thread limit per physical core.
// Essentially (depending on scheduling and other factors) # blocks fewer than HW_LIMIT can be synched, any # above is not trivially solved. cuFFT solves this.
#define HW_LIMIT (1024 / MAX_BLOCK_SIZE) * 7

__host__ void tsCombine(fftDir dir, cpx **dev_in, cpx **dev_out, int n)
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
        _kernelAll KERNEL_ARGS2(blocks, threads)(*dev_in, *dev_out, w_angle, 0xFFFFFFFF << depth, (dist - 1) << steps, steps, dist);
        cudaDeviceSynchronize();
        while (--depth > breakSize) {
            dist >>= 1;
            ++steps;
            _kernelAll KERNEL_ARGS2(blocks, threads)(*dev_out, *dev_out, w_angle, 0xFFFFFFFF << depth, (dist - 1) << steps, steps, dist);
            cudaDeviceSynchronize();
        }
        swap(dev_in, dev_out);
        ++depth;
        bSize = nBlock / 2;
        numBlocks = 1;
    }

    // Calculate complete sequence in one launch and syncronize on GPU
    _kernelGPUS KERNEL_ARGS3(blocks, threads, sizeof(cpx) * nBlock) (*dev_in, *dev_out, w_angle, w_bangle, depth, lead, breakSize, scaleCpx, numBlocks, bSize);
    cudaDeviceSynchronize();

}

// Take no usage of shared mem yet...
__global__ void _kernelAll(cpx *in, cpx *out, float angle, unsigned int lmask, unsigned int pmask, int steps, int dist)
{
    inner_k(in, out, angle, steps, lmask, pmask, dist);
}

__device__ static __inline__ void inner_k(cpx *in, cpx *out, float angle, int steps, unsigned int lmask, unsigned int pmask, int dist)
{
    cpx w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int in_low = tid + (tid & lmask);
    int in_high = in_low + dist;
    cpx in_lower = in[in_low];
    cpx in_upper = in[in_high];
    SIN_COS_F(angle * ((tid << steps) & pmask), &w.y, &w.x);
    cpx_add_sub_mul(&(out[in_low]), &(out[in_high]), in_lower, in_upper, w);    
}

__device__ static __inline__ int algorithm_c(cpx *in, cpx *out, int bit_start, int breakSize, float angle, int nBlocks, int n2)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x);
    int dist = n2;
    int steps = 0;
    init_sync(tid, nBlocks);
    inner_k(in, out, angle, steps, 0xFFFFFFFF << bit_start, (dist - 1) << steps, dist);
    __gpu_sync(nBlocks + steps);
    for (int bit = bit_start - 1; bit > breakSize; --bit) {
        dist = dist >> 1;
        ++steps;
        inner_k(out, out, angle, steps, 0xFFFFFFFF << bit, (dist - 1) << steps, dist);
        __gpu_sync(nBlocks + steps);
    }
    return breakSize + 1;
}

__device__ static __inline__ void algorithm_p(cpx *shared, int in_high, float angle, int bit)
{
    cpx w, in_lower, in_upper;
    int i = (threadIdx.x << 1);
    int ii = i + 1;
    for (int steps = 0; steps < bit; ++steps) {
        in_lower = shared[threadIdx.x];
        in_upper = shared[in_high];
        SYNC_THREADS;
        SIN_COS_F(angle * ((threadIdx.x & (0xFFFFFFFF << steps))), &w.y, &w.x);
        cpx_add_sub_mul(&(shared[i]), &(shared[ii]), in_lower, in_upper, w);
        SYNC_THREADS;
    }
}

// Full blown block syncronized algorithm! In theory this should scale up but is limited by hardware (#cores)
__global__ void _kernelGPUS(cpx *in, cpx *out, float angle, float bAngle, int depth, int lead, int breakSize, cpx scale, int nBlocks, int n2)
{
    extern __shared__ cpx shared[];
    int bit = depth;
    int in_high = n2;
    if (nBlocks > 1) {
        bit = algorithm_c(in, out, depth - 1, breakSize, angle, nBlocks, in_high);
        in_high >>= log2(nBlocks);
        in = out;
    }
    int offset = blockIdx.x * blockDim.x * 2;
    in_high += threadIdx.x;
    mem_gtos(threadIdx.x, in_high, offset, shared, in);
    algorithm_p(shared, in_high, bAngle, bit);
    mem_stog_db(threadIdx.x, in_high, offset, lead, scale, shared, out);
}