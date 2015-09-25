
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
        tmp = (double)seq[i].x;
        min_v = min(min_v, tmp);
        max_v = max(max_v, tmp);
        sum_v += tmp;
    }
    *min_val = min_v;
    *range = max_v - min_v;
    *avg = sum_v / (double)n;
}

__host__ void write_image(char *name, cpx* seq, cInt n)
{
    int x, y;
    image image;
    FILE  *fp;
    fopen_s(&fp, name, "wb");
    image = alloc_img(n, n);
    for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
            float val = (seq[y * n + x].x) * 255.f;
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
