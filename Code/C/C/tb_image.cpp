#include <Windows.h>
#include <cmath>
#include <limits>

#include "tb_image.h"
//#include "ppm.h"
#include "imglib.h"

void min_rng_avg(double *m, double *r, double *avg, cpx **com, const int n)
{
    int x, y;
    double mi, ma, mag;
    mi = DBL_MAX;
    ma = DBL_MIN;
    *avg = 0.0;
    for (y = 0; y < n; ++y)
    {
        for (x = 0; x < n; ++x)
        {
            mag = sqrt(com[y][x].r *com[y][x].r + com[y][x].i *com[y][x].i);
            mi = min(mi, mag);
            ma = max(ma, mag);
            *avg += mag;
        }
    }
    *m = mi;
    *r = ma - mi;
    *avg = *avg / (double)(n * n);
}

void cpPixel(const int px, const int px2, unsigned char *in, unsigned char *out)
{
    int p, p2;
    p = px * 3;
    p2 = px2 * 3;
    out[p] = in[p2];
    out[p + 1] = in[p2 + 1];
    out[p + 2] = in[p2 + 2];
}

void img_to_cpx(unsigned char *img, cpx **com, const int n)
{
    double r, g, b, intensity;
    int px, x, y;
    for (y = 0; y < n; ++y)
    {
        for (x = 0; x < n; ++x)
        {
            px = (y * n + x) * 3;
            r = img[px];
            g = img[px + 1];
            b = img[px + 2];
            intensity = ((r + g + b) / 3.0) / 255.0;
            com[y][x].r = (float)intensity;
            com[y][x].i = 0.f;
        }
    }
}

void cpx_to_img(cpx **com, unsigned char *img, const int n, unsigned char mag)
{
    int px, x, y;
    double magnitude, val, amin, range, avg, scale, avg_pos;

    min_rng_avg(&amin, &range, &avg, com, n);

    /* scale shift the average magnitude to avg_pos, makes for better visualizations */
    avg_pos = 0.4;
    scale = tan(avg_pos * (M_PI / 2)) / ((avg - amin) / range);

    for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
            px = (y * n + x) * 3;
            magnitude = sqrt(com[y][x].r *com[y][x].r + com[y][x].i *com[y][x].i);
            val = ((magnitude - amin) / range);
            if (mag != 0)
                val = (atan(val * scale) / (M_PI / 2.0)) * 255.0;
            else
                val *= 255.0;

            /* Convert to pixel, greyscaled... */
            img[px] = img[px + 1] = img[px + 2] = (unsigned char)(val > 255.0 ? 255 : val);
        }
    }
}

void fft_shift(unsigned char *in, unsigned char *out, const int n)
{
    int x, y, n2, px1, px2;
    n2 = n / 2;
    for (y = 0; y < n2; ++y)
    {
        for (x = 0; x < n2; ++x)
        {
            px1 = y * n + x;
            px2 = (y + n2) * n + (x + n2);
            cpPixel(px1, px2, in, out);
            cpPixel(px2, px1, in, out);
        }
    }
    for (y = 0; y < n2; ++y)
    {
        for (x = n2; x < n; ++x)
        {
            px1 = y * n + x;
            px2 = (y + n2) * n + (x - n2);
            cpPixel(px1, px2, in, out);
            cpPixel(px2, px1, in, out);
        }
    }
}

int generate_test_image_set(char *filename, char *groupname, const int size)
{
    int n, m;
    int x, y, half, px, opx, step;
    unsigned char *image, *imImage;
    char file[30];
    /* Read image to memory */
    image = readppm(filename, &n, &m);
    if (n != size || m != size)
    {
        printf("Image size not correct.\n");
        getchar();
        return 0;
    }
    if (!image)
    {
        printf("Some other error.");
        getchar();
        return 0;
    }
    step = 1;
    half = size;
    while ((half /= 2) > 2)
    {
        step *= 2;
        imImage = (unsigned char *)malloc(sizeof(unsigned char) * half * half * 3);
        for (y = 0; y < half; ++y)
        {
            for (x = 0; x < half; ++x)
            {
                px = (y * half + x) * 3;
                opx = ((y * step * half) + x) * 3 * step;
                imImage[px] = image[opx];
                imImage[px + 1] = image[opx + 1];
                imImage[px + 2] = image[opx + 2];
            }
        }
        sprintf_s(file, 30, "%s%u.ppm", groupname, half);
        writeppm(file, half, half, imImage);

        free(imImage);
    }
    return 0;
}

void writeppm(char *filename, const int width, const int height, unsigned char *img)
{
    int x, y, px;
    image image;
    FILE  *fp;
    fopen_s(&fp, filename, "wb");    
    image = alloc_img(width, height);
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            px = (y * width + x) * 3;
            put_pixel_unsafe(image, x, y, img[px + 0], img[px + 1], img[px + 2]);
        }
    }
    output_ppm(fp, image);
    fclose(fp);
}

unsigned char *readppm(char *filename, int *width, int *height)
{
    unsigned int x, y, ch, px;
    unsigned char *img;
    image image;
    color_component *cp;
    FILE *fp;
    fopen_s(&fp, filename, "rb");
    image = get_ppm(fp);
    if (!image)
        return NULL;
    img = (unsigned char *)malloc(sizeof(unsigned char) * image->width * image->height * 3);
    for (y = 0; y < image->height; ++y) {
        for (x = 0; x < image->width; ++x) {
            px = (y * image->width + x) * 3;
            cp = GET_PIXEL(image, x, y);
            for (ch = 0; ch < 3; ++ch) {
                img[px + ch] = cp[ch];
            }
        }
    }
    *width = image->width;
    *height = image->height;
    return img;
}
/*
void writeppm(char *filename, const int width, const int height, unsigned char *img)
{
    int x, y, ch, px;
    Image *image;
    image = ImageCreate(width, height);
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            px = x * y * 3;
            for (ch = 0; ch < 3; ++ch) {
                ImageSetPixel(image, x, y, ch, img[px + ch]);
            }
        }
    }
    ImageWrite(image, filename);
}

unsigned char *readppm(char *filename, int *width, int *height)
{
    int x, y, ch, px, w, h;
    unsigned char *img;
    Image *image;
    image = ImageRead(filename);
    w = ImageWidth(image);
    h = ImageHeight(image);
    img = (unsigned char *)malloc(sizeof(unsigned char) * w * h * 3);
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            px = x * y * 3;
            for (ch = 0; ch < 3; ++ch) {
                img[px + ch] = ImageGetPixel(image, x, y, ch);
            }
        }
    }
    *width = w;
    *height = h;
    return img;
}*/