
#include "tsHelper.cuh"

/* Doubtful this works... */
__host__ cudaTextureObject_t specifyTexture(cpx *dev_W)
{
    // Specify texture
    struct cudaResourceDesc resDesc;
    cudaMemset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.linear.devPtr = dev_W;
    //resDesc.res.array.array = cuArray; 

    // Specify texture object parameters 
    struct cudaTextureDesc texDesc;
    cudaMemset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object 
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}

__global__ void twiddle_factors(cpx *W, cFloat angle, cInt n)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    SIN_COS_F(angle * i, &W[i].y, &W[i].x);
}

__global__ void bit_reverse(cpx *in, cpx *out, cFloat scale, cInt lead)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p = BIT_REVERSE(i, lead);
    out[p].x = in[i].x * scale;
    out[p].y = in[i].y * scale;
}

__global__ void bit_reverse(cpx *x, cFloat dir, cInt lead, cInt n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p = BIT_REVERSE(i, lead);
    cpx tmp;
    if (i < p) {
        tmp = x[i];
        x[i] = x[p];
        x[p] = tmp;
    }
    if (dir > 0) {
        x[i].x = x[i].x / (float)n;
        x[i].y = x[i].y / (float)n;
    }
}

__host__ void set2DBlocksNThreads(dim3 *bFFT, dim3 *tFFT, dim3 *bTrans, dim3 *tTrans, cInt n)
{
    cInt n2 = n >> 1;
    (*bFFT).x = n;
    (*bFFT).z = (*tFFT).y = (*tFFT).z = (*bTrans).z = (*tTrans).z = 1;
    (*bTrans).x = (*bTrans).y = (n / TILE_DIM);
    (*tTrans).x = (*tTrans).y = THREAD_TILE_DIM;
    if (n2 > MAX_BLOCK_SIZE) {
        (*bFFT).y = n2 / MAX_BLOCK_SIZE;
        (*tFFT).x = MAX_BLOCK_SIZE;
    }
    else {
        (*bFFT).y = 1;
        (*tFFT).x = n2;
    }
}

__host__ void checkCudaError()
{
    cudaError_t e;
    if (e = cudaGetLastError()) printf("%s: %s\n", cudaGetErrorName(e), cudaGetErrorString(e));
}

__host__ cpx* read_image(char *name, int *n)
{
    image image;
    color_component *cp;
    FILE *fp;
    fopen_s(&fp, name, "rb");
    image = get_ppm(fp);
    if (!image || image->width != image->height)
        return NULL;

    int size = *n = image->width;
    cpx *seq = (cpx *)malloc(sizeof(cpx) * size * size);
    for (int y = 0; y < (int)image->height; ++y) {
        for (int x = 0; x < (int)image->width; ++x) {
            cp = GET_PIXEL(image, x, y);
            seq[y * size + x] = make_cuComplex((cp[0] + cp[1] + cp[2]) / (3.f * 255.f), 0.f);
        }
    }
    free_img(image);
    return seq;
}

__host__ void normalized_cpx_values(cpx* seq, cInt n, double *min_val, double *range, double *avg)
{
    double min_v = DBL_MAX;
    double max_v = DBL_MIN;
    double sum_v = 0.0;
    double tmp = 0.0;
    for (int i = 0; i < n; ++i) {
        tmp = cuCabsf(seq[i]);
        min_v = min(min_v, tmp);
        max_v = max(max_v, tmp);
        sum_v += tmp;
    }
    *min_val = min_v;
    *range = max_v - min_v;
    *avg = sum_v / (double)n;
}

__host__ void write_normalized_image(char *name, cpx* seq, cInt n)
{
    image image;
    FILE  *fp;
    double minVal, range, avg, mag, val;
    normalized_cpx_values(seq, n, &minVal, &range, &avg);
    double avg_pos = 0.4;
    double scale = tan(avg_pos * (M_PI / 2)) / ((avg - minVal) / range);
    image = alloc_img(n, n);
    fopen_s(&fp, name, "wb");
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            mag = cuCabsf(seq[y * n + x]);
            val = ((mag - minVal) / range);
            val = (atan(val * scale) / (M_PI / 2.0)) * 255.0;
            color_component col = (unsigned char)(val > 255.0 ? 255 : val);
            put_pixel_unsafe(image, x, y, col, col, col);
        }
    }
    output_ppm(fp, image);
    fclose(fp);
    free_img(image);
}

__host__ void normalized_image(cpx* seq, cInt n)
{
    double minVal, range, avg, mag, val;
    normalized_cpx_values(seq, n, &minVal, &range, &avg);
    double avg_pos = 0.8;
    double scale = tan(avg_pos * (M_PI / 2)) / ((avg - minVal) / range);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            mag = cuCabsf(seq[y * n + x]);
            val = ((mag - minVal) / range);
            val = (atan(val * scale) / (M_PI / 2.0));
            seq[y * n + x] = make_cuFloatComplex((float)(val > 1.0 ? 1 : val), 0.f);
        }
    }
}

__host__ void write_image(char *name, char *type, cpx* seq, cInt n)
{
    image image;
    FILE  *fp;
    image = alloc_img(n, n);
    char filename[50];
    sprintf_s(filename, 50, "out/img/%s_%u_%s.ppm", name, n, type);
    fopen_s(&fp, filename, "wb");
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            color_component val = (unsigned char)((seq[y * n + x].x) * 255.f);
            put_pixel_unsafe(image, x, y, val, val, val);
        }
    }
    output_ppm(fp, image);
    fclose(fp);
    free_img(image);
}

__host__ void clear_image(cpx* seq, cInt n)
{
    for (int i = 0; i < n; ++i)
        seq[i] = make_cuFloatComplex(1.f, 1.f);
}

__host__ void cpPixel(cInt px, cInt px2, cCpx *in, cpx *out)
{
    int p, p2;
    p = px * 3;
    p2 = px2 * 3;
    out[p] = in[p2];
    out[p + 1] = in[p2 + 1];
    out[p + 2] = in[p2 + 2];
}

__host__ cpx* fftShift(cpx *seq, cInt n)
{
    cpx *out = (cpx *)malloc(sizeof(cpx)*n*n);
    int px1, px2;
    int n2 = n / 2;
    for (int y = 0; y < n2; ++y) {
        for (int x = 0; x < n2; ++x) {
            px1 = y * n + x;
            px2 = (y + n2) * n + (x + n2);
            out[px1] = seq[px2];
            out[px2] = seq[px1];
        }
    }
    for (int y = 0; y < n2; ++y) {
        for (int x = n2; x < n; ++x) {
            px1 = y * n + x;
            px2 = (y + n2) * n + (x - n2);
            out[px1] = seq[px2];
            out[px2] = seq[px1];
        }
    }
    return out;
}
