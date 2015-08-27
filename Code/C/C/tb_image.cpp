#include <Windows.h>
#include <cmath>
#include <limits>

#include "tb_image.h"

/* These are not written for performance, only visualizations and conversions.
 */

void min_rng_avg(double *m, double *r, double *avg, tb_cpx **com, const int n)
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

void img_to_cpx(unsigned char *img, tb_cpx **com, const int n)
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

void cpx_to_img(tb_cpx **com, unsigned char *img, const int n, unsigned char mag)
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
                opx = ((y * step * half) + x ) * 3 * step;
                imImage[px] = image[opx];
                imImage[px + 1] = image[opx + 1];
                imImage[px + 2] = image[opx + 2];
            }
        }        
        sprintf_s(file, 30, "%s_%u.ppm", groupname, half);
        writeppm(file, half, half, imImage);

        free(imImage);
    }
}

int writeppm(char *filename, int width, int height, unsigned char *data)
{
    FILE *fp;
    int error = 1;
    int i, h, v;

    if (filename != NULL)
    {
        fopen_s(&fp, filename, "w");

        if (fp != NULL)
        {
            // Write PPM file
            // Header	
            fprintf(fp, "P3\n");
            fprintf(fp, "# written by Ingemars PPM writer\n");
            fprintf(fp, "%d %d\n", width, height);
            fprintf(fp, "%d\n", 255); // range

            // Data
            for (v = height - 1; v >= 0; v--)
            {
                for (h = 0; h < width; h++)
                {
                    i = (width*v + h) * 3; // assumes rgb, not rgba
                    fprintf(fp, "%d %d %d ", data[i], data[i + 1], data[i + 2]);
                }
                fprintf(fp, "\n"); // range
            }

            if (fwrite("\n", sizeof(char), 1, fp) == 1)
                error = 0; // Probable success
            fclose(fp);
        }
    }
    return(error);
}

unsigned char *readppm(char *filename, int *width, int *height)
{
    FILE *fd;
    int  k;//, nm;
    char c;
    int i, j;
    char b[100];
    //float s;
    int red, green, blue;
    long numbytes;//, howmuch;
    int n;
    int m;
    unsigned char *image;

    fopen_s(&fd, filename, "rb");
    if (fd == NULL)
    {
        printf("Could not open %s\n", filename);
        return NULL;
    }
    c = getc(fd);
    if (c == 'P' || c == 'p')
        c = getc(fd);

    if (c == '3')
    {
        //printf("%s is a PPM file (plain text version)\n", filename);

        // NOTE: This is not very good PPM code! Comments are not allowed
        // except immediately after the magic number.
        c = getc(fd);
        if (c == '\n' || c == '\r') // Skip any line break and comments
        {
            c = getc(fd);
            while (c == '#')
            {
                fscanf_s(fd, "%[^\n\r] ", b);
                printf("%s\n", b);
                c = getc(fd);
            }
            ungetc(c, fd);
        }
        fscanf_s(fd, "%d %d %d", &n, &m, &k);

        //printf("%d rows  %d columns  max value= %d\n", n, m, k);

        numbytes = n * m * 3;
        image = (unsigned char *)malloc(numbytes);
        if (image == NULL)
        {
            printf("Memory allocation failed!\n");
            return NULL;
        }
        for (i = m - 1; i >= 0; i--) for (j = 0; j < n; j++) // Important bug fix here!
        { // i = row, j = column
            fscanf_s(fd, "%d %d %d", &red, &green, &blue);
            image[(i*n + j) * 3] = red * 255 / k;
            image[(i*n + j) * 3 + 1] = green * 255 / k;
            image[(i*n + j) * 3 + 2] = blue * 255 / k;
        }
    }
    else
        if (c == '6')
        {
            //printf("%s is a PPM file (raw version)!\n", filename);

            c = getc(fd);
            if (c == '\n' || c == '\r') // Skip any line break and comments
            {
                c = getc(fd);
                while (c == '#')
                {
                    fscanf_s(fd, "%[^\n\r] ", b);
                    printf("%s\n", b);
                    c = getc(fd);
                }
                ungetc(c, fd);
            }
            fscanf_s(fd, "%d %d %d", &n, &m, &k);
            //printf("%d rows  %d columns  max value= %d\n", m, n, k);
            c = getc(fd); // Skip the last whitespace

            numbytes = n * m * 3;
            image = (unsigned char *)malloc(numbytes);
            if (image == NULL)
            {
                printf("Memory allocation failed!\n");
                return NULL;
            }
            // Read and re-order as necessary
            for (i = m - 1; i >= 0; i--) for (j = 0; j < n; j++) // Important bug fix here!
            {
                image[(i*n + j) * 3 + 0] = getc(fd);
                image[(i*n + j) * 3 + 1] = getc(fd);
                image[(i*n + j) * 3 + 2] = getc(fd);
            }
        }
        else
        {
            printf("%s is not a PPM file!\n", filename);
            return NULL;
        }

    //printf("read image\n");

    *height = m;
    *width = n;
    return image;
}