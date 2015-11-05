#include "imglib.h"

#define PPMREADBUFLEN 256

struct image_t 
{
    unsigned int width;
    unsigned int height;
    pixel * buf;
};

image get_ppm(FILE *pf)
{
    char buf[PPMREADBUFLEN], *t;
    image img;
    unsigned int w, h, d;
    int r;

    if (pf == NULL) return NULL;
    t = fgets(buf, PPMREADBUFLEN, pf);
    /* the code fails if the white space following "P6" is not '\n' */
    if ((t == NULL) || (strncmp(buf, "P6\n", 3) != 0)) return NULL;
    do
    { /* Px formats can have # comments after first line */
        t = fgets(buf, PPMREADBUFLEN, pf);
        if (t == NULL) return NULL;
    } while (strncmp(buf, "#", 1) == 0);
    r = sscanf_s(buf, "%u %u", &w, &h);
    if (r < 2) return NULL;

    r = fscanf_s(pf, "%u", &d);
    if ((r < 1) || (d != 255)) return NULL;
    fseek(pf, 1, SEEK_CUR); /* skip one byte, should be whitespace */

    img = alloc_img(w, h);
    if (img != NULL)
    {
        size_t rd = fread(img->buf, sizeof(pixel), w*h, pf);
        if (rd < w*h)
        {
            free_img(img);
            return NULL;
        }
        return img;
    }
    return NULL;
}

void output_ppm(FILE *fd, image img)
{
    unsigned int n;
    (void)fprintf(fd, "P6\n%d %d\n255\n", img->width, img->height);
    n = img->width * img->height;
    (void)fwrite(img->buf, sizeof(pixel), n, fd);
    (void)fflush(fd);
}

image alloc_img(unsigned int width, unsigned int height)
{
    int size;
    image img;
    size = width * height * sizeof(pixel);
    img = (image)malloc(sizeof(image_t));
    img->buf = (pixel *)malloc(size);
    img->width = width;
    img->height = height;
    return img;
}

void free_img(image img)
{
    free(img->buf);
    free(img);
}

void fill_img(
    image img,
    color_component r,
    color_component g,
    color_component b)
{
    unsigned int i, n;
    n = img->width * img->height;
    for (i = 0; i < n; ++i)
    {
        img->buf[i][0] = r;
        img->buf[i][1] = g;
        img->buf[i][2] = b;
    }
}

void put_pixel_unsafe(
    image img,
    unsigned int x,
    unsigned int y,
    color_component r,
    color_component g,
    color_component b)
{
    unsigned int ofs;
    ofs = (y * img->width) + x;
    img->buf[ofs][0] = r;
    img->buf[ofs][1] = g;
    img->buf[ofs][2] = b;
}

void put_pixel_clip(
    image img,
    unsigned int x,
    unsigned int y,
    color_component r,
    color_component g,
    color_component b)
{
    if (x < img->width && y < img->height)
        put_pixel_unsafe(img, x, y, r, g, b);
}

//
// My own additions to handle my custom format
//

void read_image(cpx *dst, char *name, int *n)
{
    image image;
    color_component *cp;
    FILE *fp;
    fopen_s(&fp, name, "rb");
    image = get_ppm(fp);
    if (!image || image->width != image->height)
        return;

    int size = *n = image->width;
    for (int y = 0; y < (int)image->height; ++y) {
        for (int x = 0; x < (int)image->width; ++x) {
            cp = GET_PIXEL(image, x, y);
            dst[y * size + x] = make_cuComplex((cp[0] + cp[1] + cp[2]) / (3.f * 255.f), 0.f);
        }
    }
    free_img(image);
}

cpx *get_flat(cpx **seq, int n)
{
    cpx *flat = (cpx *)malloc(sizeof(cpx) * n * n);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            flat[y * n + x] = seq[y][x];
    return flat;
}

void write_image(char *name, char *type, cpx* seq, int n)
{
    image image;
    image = alloc_img(n, n);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            color_component val = (unsigned char)((seq[y * n + x].x) * 255.f);
            put_pixel_unsafe(image, x, y, val, val, val);
        }
    }
    FILE *fp = get_img_file_pntr(name, n, type);
    output_ppm(fp, image);
    fclose(fp);
    free_img(image);
}

void write_image(char *name, char *type, cpx** seq, int n)
{
    cpx *flat = get_flat(seq, n);
    write_image(name, type, flat, n);
    free(flat);
}

void normalized_cpx_values(cpx* seq, int n, double *min_val, double *range, double *average_best)
{
    double min_v = 99999999999;
    double max_v = -99999999999;
    double sum_v = 0.0;
    double tmp = 0.0;
    for (int i = 0; i < n; ++i) {
        tmp = cuCabsf(seq[i]);        
        min_v = min_v < tmp ? min_v : tmp;
        max_v = max_v > tmp ? max_v : tmp;
        sum_v += tmp;
    }
    *min_val = min_v;
    *range = max_v - min_v;
    *average_best = sum_v / (double)n;
}

void write_normalized_image(char *name, char *type, cpx* seq, int n, bool doFFTShift)
{
    image image;
    double minVal, range, average_best, mag, val;
    cpx *tmp = 0;
    if (doFFTShift) {
        tmp = (cpx *)malloc(sizeof(cpx) * n * n);
        fftShift(tmp, seq, n);
    }
    normalized_cpx_values(tmp, n, &minVal, &range, &average_best);
    double avg_pos = 0.1;
    double scalar = tan(avg_pos * (M_PI / 2.0)) / ((average_best - minVal) / range);
    image = alloc_img(n, n);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            mag = cuCabsf(tmp[y * n + x]);
            val = ((mag - minVal) / range);
            val = (atan(val * scalar) / (M_PI / 2.0)) * 255.0;
            color_component col = (unsigned char)(val > 255.0 ? 255 : val);
            put_pixel_unsafe(image, x, y, col, col, col);
        }
    }
    FILE *fp = get_img_file_pntr(name, n, type);
    output_ppm(fp, image);
    fclose(fp);
    free_img(image);
    if (tmp) {
        free(tmp);
    }
}

void write_normalized_image(char *name, char *type, cpx** seq, int n, bool doFFTShift)
{
    cpx *flat = get_flat(seq, n);    
    write_normalized_image(name, type, flat, n, doFFTShift);
    free(flat);
}

void normalized_image(cpx* seq, int n)
{
    double minVal, range, average_best, mag, val;
    normalized_cpx_values(seq, n, &minVal, &range, &average_best);
    double avg_pos = 0.8;
    double scalar = tan(avg_pos * (M_PI / 2.0)) / ((average_best - minVal) / range);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            mag = cuCabsf(seq[y * n + x]);
            val = ((mag - minVal) / range);
            val = (atan(val * scalar) / (M_PI / 2.0));
            seq[y * n + x] = make_cuFloatComplex((float)(val > 1.0 ? 1 : val), 0.f);
        }
    }
}

void clear_image(cpx* seq, int n)
{
    for (int i = 0; i < n; ++i)
        seq[i] = make_cuFloatComplex(1.f, 1.f);
}

void cpPixel(int px, int px2, cpx *in, cpx *out)
{
    int p, p2;
    p = px * 3;
    p2 = px2 * 3;
    out[p] = in[p2];
    out[p + 1] = in[p2 + 1];
    out[p + 2] = in[p2 + 2];
}

void fftShift(cpx *dst, cpx *src, int n)
{
    int px1, px2;
    int n_half = n / 2;
    for (int y = 0; y < n_half; ++y) {
        for (int x = 0; x < n_half; ++x) {
            px1 = y * n + x;
            px2 = (y + n_half) * n + (x + n_half);
            dst[px1] = src[px2];
            dst[px2] = src[px1];
        }
    }
    for (int y = 0; y < n_half; ++y) {
        for (int x = n_half; x < n; ++x) {
            px1 = y * n + x;
            px2 = (y + n_half) * n + (x - n_half);
            dst[px1] = src[px2];
            dst[px2] = src[px1];
        }
    }
}