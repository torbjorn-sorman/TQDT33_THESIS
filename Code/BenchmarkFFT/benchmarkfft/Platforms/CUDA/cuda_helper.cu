#include "cuda_helper.cuh"
#if defined(_NVIDIA)
void checkCudaError(char *msg)
{
    cudaError_t e;
    if (e = cudaGetLastError()) printf("%s:\n%s: %s\n", msg, cudaGetErrorName(e), cudaGetErrorString(e));
}

void checkCudaError()
{
    cudaError_t e;
    if (e = cudaGetLastError()) printf("%s: %s\n", cudaGetErrorName(e), cudaGetErrorString(e));
}

#define ERROR_MARGIN 0.0001

// Useful functions for debugging
void console_print(cpx *seq, int n)
{
    for (int i = 0; i < n; ++i) printf("%f\t%f\n", seq[i].x, seq[i].y);
}

void console_print_cpx_img(cpx *seq, int n)
{
    printf("\n");
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            printf("%.2f\t", seq[y * n + x].x);
        }
        printf("\n");
    }
    printf("\n");
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            printf("%.2f\t", seq[y * n + x].y);
        }
        printf("\n");
    }
}

void cuda_setup_buffers(int n, cpx **dev_in, cpx **dev_out, cpx **in, cpx **ref, cpx **out)
{
    size_t total_size = batch_size(n);
    int batches = batch_count(n);
    if (dev_in)  { *dev_in = 0;  cudaMalloc((void**)dev_in,  total_size * sizeof(cpx)); }
    if (dev_out) { *dev_out = 0; cudaMalloc((void**)dev_out, total_size * sizeof(cpx)); }    
    if (in) *in =   get_seq(n, batches, 1);
    if (ref) *ref = get_seq(n * batches, *in);
    if (out) *out = get_seq(n * batches);
}

void _cudaFree(cpx **dev_in, cpx **dev_out)
{
    cudaFree(*dev_in);
    if (dev_out != NULL)
        cudaFree(*dev_out);
}

int cuda_shakedown(int n, cpx **dev_in, cpx **dev_out, cpx **in, cpx **ref, cpx **out)
{
    _cudaFree(dev_in, dev_out);
    cudaDeviceSynchronize();
    double diff;
    if (in && ref) {
        diff = diff_seq(*in, *ref, batch_size(n));
        free_all(*in, *ref);
    }
    if (out) {
        free(*out);
    }
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return diff > ERROR_MARGIN;
}

void cuda_setup_buffers_2d(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o, size_t *size, int n)
{
    char input_file[40];
    sprintf_s(input_file, 40, "Images/%u.ppm", n);
    int sz;
    size_t total_size = batch_size(n * n) * sizeof(cpx);

    *in = (cpx *)malloc(total_size); 
    read_image(*in, input_file, &sz);
    for (int i = 1; i < batch_count(n * n); ++i) {
        memcpy(*in + i * n * n, *in, n * n);
    }
    *ref = (cpx *)malloc(total_size);
    memcpy(*ref, *in, total_size);
    *size = total_size;
    cudaMalloc((void**)dev_i, total_size);
    cudaMalloc((void**)dev_o, total_size);
    cudaMemcpy(*dev_i, *in, total_size, cudaMemcpyHostToDevice);
}

void cuda_shakedown_2d(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o)
{    
    free_all(*in, *ref);
    cudaFree(*dev_i);
    cudaFree(*dev_o);
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

int cuda_compare_result(cpx *in, cpx *ref, cpx *dev, size_t size, int len)
{
    cudaMemcpy(in, dev, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; ++i) {
        if (cuCabsf(cuCsubf(in[i], ref[i])) > 0.0001) {
            return 0;
        }
    }
    return 1;
}

#define max(a, b) ((a) > (b) ? (a) : (b))

int cuda_compare_result(cpx *in, cpx *ref, cpx *dev, size_t size, int len, double *relDiff)
{
    double mDiff = 0.0;
    double mVal = -1;
    cudaMemcpy(in, dev, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; ++i) {
        mVal = max(mVal, max(cuCabsf(in[i]), cuCabsf(ref[i])));
        double tmp = cuCabsf(cuCsubf(in[i], ref[i]));
        mDiff = tmp > mDiff ? tmp : mDiff;
    }
    *relDiff = (mDiff / mVal);
    return *relDiff < 0.00001;
}

void cudaCheckError(cudaError_t err)
{
    if (err) {
        printf("\n%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        getchar();
        exit(err);
    }
}

void cudaCheckError()
{
    cudaCheckError(cudaGetLastError());
}
#endif