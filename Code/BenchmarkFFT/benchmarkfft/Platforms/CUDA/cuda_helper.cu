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

void _cudaMalloc(int n, cpx **dev_in, cpx **dev_out)
{
    *dev_in = 0;    
    cudaMalloc((void**)dev_in, n * sizeof(cpx));
    if (dev_out != NULL) {
        *dev_out = 0;
        cudaMalloc((void**)dev_out, n * sizeof(cpx));
    }
}

void cuda_setup_buffers(int n, cpx **dev_in, cpx **dev_out, cpx **in, cpx **ref, cpx **out)
{
    _cudaMalloc(n, dev_in, dev_out);
    if (in == NULL && ref == NULL && out == NULL)
        return;
    fft_alloc_sequences(n, in, ref, out);
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
    if (in == NULL && ref == NULL && out == NULL)
        return 0;
    double diff = diff_seq(*in, *ref, n);
    free_all(*in, *out, *ref);
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
    *in = (cpx *)malloc(sizeof(cpx) * n * n); 
    read_image(*in, input_file, &sz);
    *ref = (cpx *)malloc(sizeof(cpx) * n * n);
    memcpy(*ref, *in, sizeof(cpx) * n * n);
    *size = n * n * sizeof(cpx);
    cudaMalloc((void**)dev_i, *size);
    cudaMalloc((void**)dev_o, *size);
    cudaMemcpy(*dev_i, *in, *size, cudaMemcpyHostToDevice);
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