#include "imglib.h"

#define PPMREADBUFLEN 256

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

cpx* read_image(char *name, int *n)
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