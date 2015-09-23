#include <stdio.h>
#include <device_launch_parameters.h>

#include "tsHelper.cuh"
#include "math.h"

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

__global__ void twiddle_factors(cpx *W, const float angle, const int n)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    SIN_COS_F(angle * i, &W[i].y, &W[i].x);
}

__global__ void bit_reverse(cpx *in, cpx *out, const float scale, const int lead)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int p = BIT_REVERSE(i, lead);
    out[p].x = in[i].x * scale;
    out[p].y = in[i].y * scale;
}

__global__ void bit_reverse(cpx *x, const float dir, const int lead, const int n)
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

__host__ cpx* read_image(char *name, int *n)
{
    unsigned int ch, px;
    unsigned char *img;
    image image;
    color_component *cp;
    FILE *fp;
    fopen_s(&fp, name, "rb");
    image = get_ppm(fp);
    if (!image || image->width != image->height)
        return NULL;

    int size = *n = image->width;
    cpx *seq = (cpx *)malloc(sizeof(cpx) * size * size);
    for (int y = 0; y < image->height; ++y) {
        for (int x = 0; x < image->width; ++x) {            
            cp = GET_PIXEL(image, x, y);
            seq[y * size + x] = make_cuComplex((cp[0] + cp[1] + cp[2]) / 3.f, 0.f);
        }
    }
    free_img(image);
    return seq;
}

__host__ void write_image(char *name, cpx* seq, const int n)
{
    int x, y, px;
    image image;
    FILE  *fp;
    fopen_s(&fp, name, "wb");
    image = alloc_img(n, n);
    for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
            px = (y * n + x) * 3;
            float val = seq[y * n + x].x;
            put_pixel_unsafe(image, x, y, val, val, val);
        }
    }
    output_ppm(fp, image);
    fclose(fp);
}