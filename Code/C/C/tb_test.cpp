#include <Windows.h>
#include <limits>

#include "tb_test.h"
#include "tb_fft.h"
#include "tb_image.h"
#include "tb_transpose.h"
#include "tb_math.h"
#include "tb_print.h"
#include <algorithm>

#define NO_TESTS 16
#define ERROR_MARGIN 0.0002

int checkError(tb_cpx *seq, tb_cpx *ref, const int n, int print)
{
    int j;
    double r, i, i_val, r_val;
    r = i = 0.0;
    for (j = 0; j < j; ++j)
    {
        r_val = abs(seq[j].r - ref[j].r);
        i_val = abs(seq[j].i - ref[j].i);
        r = r > r_val ? r : r_val;
        i = i > i_val ? i : i_val;
    }
    if (print == 1) printf("Error\tre(e): %f\t im(e): %f\t\t\t%u\n", r, i, n);
    return r > ERROR_MARGIN || i > ERROR_MARGIN;
}

int cmp(const void *x, const void *y)
{
    double xx = *(double*)x, yy = *(double*)y;
    if (xx < yy) return -1;
    if (xx > yy) return  1;
    return 0;
}

double avg(double *m, int n)
{
    int i, cnt;
    double sum;
    qsort(m, n, sizeof(double), cmp);
    sum = 0.0;
    cnt = 0;

    for (i = (n / 2); i < n; ++i) {
        sum += m[i];
        ++cnt;
    }
    return (sum / cnt);
}

void simple()
{
    const int n = 16;
    int i;
    tb_cpx *in, *in2, *out;
    in = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    in2 = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    out = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    for (i = 0; i < n; ++i)
    {
        in[i].r = 0.f;//(float)sin(M_2_PI * (((double)i) / n));
        in[i].i = 0.f;
        in2[i] = in[i];
    }
    in[1].r = 1.f;
    in2[1] = in[1];
    console_print(in, n);
    console_separator(1);
    tb_fft(FORWARD_FFT, in2, in2, n);
    console_print(out, n);
    console_separator(1);
    console_print(in2, n);
}

unsigned char test_equal_dft(fft_function fft_fn, fft_function dft_ref_fn, const int inplace)
{
    int i, n;
    unsigned char res = 1;
    tb_cpx *fft_out, *dft_out, *in, *in2;

    for (n = 2; n < 4194304; n *= 2) {
        in = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
        for (i = 0; i < n; ++i)
        {
            in[i].r = (float)sin(M_2_PI * (((double)i) / n));
            in[i].i = 0.f;
        }
        if (inplace == 0)
        {
            fft_out = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
            dft_out = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
            dft_ref_fn(FORWARD_FFT, in, dft_out, n);
            fft_fn(FORWARD_FFT, in, fft_out, n);
            res = checkError(dft_out, fft_out, n, 1);
            free(dft_out);
            free(fft_out);
        }
        else
        {
            in2 = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
            for (i = 0; i < n; ++i) {
                in2[i].r = in[i].r;
                in2[i].i = in[i].i;
            }
            dft_ref_fn(FORWARD_FFT, in, in, n);
            fft_fn(FORWARD_FFT, in2, in2, n);
            res = checkError(in, in2, n, 1);
            free(in2);
        }
        free(in);
    }
    return res;
}

unsigned char test_equal_dft2d(fft_function fft_fn, fft_function ref_fn, const int inplace)
{
    int n, m, i, size;
    unsigned char *image, *imImage, *imImageRef;
    tb_cpx **cpxImg, **cpxImgRef;

    size = 512;
    image = readppm("lena_512.ppm", &n, &m);
    imImage = (unsigned char *)malloc(sizeof(unsigned char) * size * size * 3);
    imImageRef = (unsigned char *)malloc(sizeof(unsigned char) * size * size * 3);
    cpxImg = (tb_cpx **)malloc(sizeof(tb_cpx) * size);
    cpxImgRef = (tb_cpx **)malloc(sizeof(tb_cpx) * size);
    for (i = 0; i < size; ++i) {
        cpxImg[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * size);
        cpxImgRef[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * size);
    }
    img_to_cpx(image, cpxImg, size);
    img_to_cpx(image, cpxImgRef, size);

    tb_fft2d(FORWARD_FFT, fft_fn, cpxImg, size);
    tb_fft2d(FORWARD_FFT, ref_fn, cpxImgRef, size);

    printf("Max Fw error: %f\n", cpx_diff(cpxImg, cpxImgRef, size));
    printf("Avg Fw error: %f\n", cpx_avg_diff(cpxImg, cpxImgRef, size));

    tb_fft2d(INVERSE_FFT, fft_fn, cpxImg, size);
    tb_fft2d(INVERSE_FFT, ref_fn, cpxImgRef, size);

    printf("Max Inv error: %f\n", cpx_diff(cpxImg, cpxImgRef, size));
    printf("Avg Inv error: %f\n", cpx_avg_diff(cpxImg, cpxImgRef, size));

    cpx_to_img(cpxImg, imImage, size, 0);
    cpx_to_img(cpxImgRef, imImageRef, size, 0);

    writeppm("test_equal.ppm", size, size, imImage);
    writeppm("test_equal_ref.ppm", size, size, imImageRef);

    free(image);
    for (i = 0; i < size; ++i) {
        free(cpxImg[i]);
        free(cpxImgRef[i]);
    }
    free(cpxImg);
    free(cpxImgRef);
    free(imImage);
    free(imImageRef);
    return 1;
}

double test_time_dft(fft_function fft_fn, const int n)
{
    int tests = (NO_TESTS * 4 / log2_32(n));
    LARGE_INTEGER freq, tStart, tStop;
    double *measures, retval;
    measures = (double *)malloc(sizeof(double) * tests);
    tb_cpx *in = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    tb_cpx *out = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    int i;
    for (i = 0; i < n; ++i) {
        in[i].r = (float)sin(M_2_PI * ((double)i / n));
        in[i].i = 0.f;
    }
    QueryPerformanceFrequency(&freq);
    for (i = 0; i < tests; ++i) {
        QueryPerformanceCounter(&tStart);
        fft_fn(FORWARD_FFT, in, out, n);
        QueryPerformanceCounter(&tStop);
        measures[i] = (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart;
    }
    free(in);
    free(out);
    retval = avg(measures, tests);
    free(measures);
    return retval;
}

double test_time_dft_2d(fft_function fft_fn, const int n)
{
    int x, y, i, k, l;
    LARGE_INTEGER freq, tStart, tStop;
    char filename[30];
    unsigned char *image;
    tb_cpx **cpxImg, **cpxImgRef;
    double measures[NO_TESTS];

    cpxImg = (tb_cpx **)malloc(sizeof(tb_cpx) * n);
    cpxImgRef = (tb_cpx **)malloc(sizeof(tb_cpx) * n);
    for (i = 0; i < n; ++i) {
        cpxImg[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
        cpxImgRef[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    }
    sprintf_s(filename, 30, "lena_%u.ppm", n);
    image = readppm(filename, &x, &y);
    img_to_cpx(image, cpxImgRef, n);

    QueryPerformanceFrequency(&freq);
    for (i = 0; i < NO_TESTS; ++i)
    {
        for (k = 0; k < n; ++k) {
            for (l = 0; l < n; ++l) {
                cpxImg[k][l].r = cpxImgRef[k][l].r;
                cpxImg[k][l].i = cpxImgRef[k][l].i;
            }
        }
        QueryPerformanceCounter(&tStart);
        tb_fft2d(FORWARD_FFT, fft_fn, cpxImg, n);
        tb_fft2d(INVERSE_FFT, fft_fn, cpxImg, n);
        QueryPerformanceCounter(&tStop);
        measures[i] = (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart;
    }
    free(image);
    for (i = 0; i < n; ++i)
        free(cpxImg[i]);
    free(cpxImg);
    return measures[NO_TESTS];
}

double test_time_2d(const int openMP)
{
    const int n = 512;
    int x, y, i, k, l;
    LARGE_INTEGER freq, tStart, tStop;
    char filename[30];
    unsigned char *image;
    tb_cpx **cpxImg, **cpxImgRef;
    double measures[NO_TESTS];

    cpxImg = (tb_cpx **)malloc(sizeof(tb_cpx) * n);
    cpxImgRef = (tb_cpx **)malloc(sizeof(tb_cpx) * n);
    for (i = 0; i < n; ++i) {
        cpxImg[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
        cpxImgRef[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    }
    sprintf_s(filename, 30, "lena_%u.ppm", n);
    image = readppm(filename, &x, &y);
    img_to_cpx(image, cpxImgRef, n);

    QueryPerformanceFrequency(&freq);
    for (i = 0; i < NO_TESTS; ++i)
    {
        for (k = 0; k < n; ++k) {
            for (l = 0; l < n; ++l) {
                cpxImg[k][l].r = cpxImgRef[k][l].r;
                cpxImg[k][l].i = cpxImgRef[k][l].i;
            }
        }
        if (openMP == 1) {
            QueryPerformanceCounter(&tStart);
            tb_fft2d_openmp(FORWARD_FFT, tb_fft, cpxImg, n);
            tb_fft2d_openmp(INVERSE_FFT, tb_fft, cpxImg, n);
            QueryPerformanceCounter(&tStop);
        }
        else {
            QueryPerformanceCounter(&tStart);
            tb_fft2d(FORWARD_FFT, tb_fft, cpxImg, n);
            tb_fft2d(INVERSE_FFT, tb_fft, cpxImg, n);
            QueryPerformanceCounter(&tStop);
        }
        measures[i] = (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart;
    }
    free(image);
    for (i = 0; i < n; ++i)
        free(cpxImg[i]);
    free(cpxImg);
    return avg(measures, NO_TESTS);
}

double test_cmp_time(fft_function fn, fft_function ref)
{
    int n;
    double time, time_ref, diff, rel, sum, sum_ref;
    rel = DBL_MIN;
    sum = sum_ref = 0.0;
    printf("\trel.\tdiff.\ttime\tref\tN\n");
    time = test_time_dft(fn, 512);
    time_ref = test_time_dft(ref, 512);
    for (n = 8; n < 8388608; n *= 2) {
        time = test_time_dft(fn, n);
        time_ref = test_time_dft(ref, n);
        diff = time_ref - time;
        sum += time;
        sum_ref += time_ref;
        rel += diff / time_ref;
        printf("(ms)\t%.2f\t%.3f\t%.3f\t%.3f\t%u\n", diff / time_ref, diff, time, time_ref, n);

    }
    return rel / 22;
}

unsigned char test_image(fft_function fft_fn, char *filename, const int n)
{
    int res, w, m, i;
    char file[30];
    unsigned char *image, *imImage, *imImage2, *greyImage;
    tb_cpx **cpxImg;
    sprintf_s(file, 30, "%s_%u.ppm", filename, n);
    printf("Read: %s\n", file);
    image = readppm(file, &w, &m);
    if (w != n || m != n)
    {
        printf("Image size not square and pw of 2.\n");
        getchar();
        return 0;
    }
    greyImage = (unsigned char *)malloc(sizeof(unsigned char) * n * n * 3);
    imImage = (unsigned char *)malloc(sizeof(unsigned char) * n * n * 3);
    imImage2 = (unsigned char *)malloc(sizeof(unsigned char) * n * n * 3);
    cpxImg = (tb_cpx **)malloc(sizeof(tb_cpx) * n);
    for (i = 0; i < n; ++i)
    {
        cpxImg[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    }

    /* Set real values from image values.
    * Store the real-value version of the image.
    */
    img_to_cpx(image, cpxImg, n);
    writeppm("img00-org.ppm", n, n, image);
    cpx_to_img(cpxImg, greyImage, n, 0);
    printf("Write img00-grey.ppm\n");
    writeppm("img00-grey.ppm", n, n, greyImage);
    /* Run 2D FFT on complex values.
    * Map absolute values of complex to pixels and store to file.
    */
    tb_fft2d(FORWARD_FFT, tb_fft, cpxImg, n);
    cpx_to_img(cpxImg, imImage, n, 1);
    fft_shift(imImage, imImage2, n);
    printf("Write img01-magnitude.ppm\n");
    writeppm("img01-magnitude.ppm", n, n, imImage2);

    /* Run inverse 2D FFT on complex values */
    tb_fft2d(INVERSE_FFT, tb_fft, cpxImg, n);
    cpx_to_img(cpxImg, imImage, n, 0);
    printf("Write img02-fftToImage.ppm\n");
    writeppm("img02-fftToImage.ppm", n, n, imImage);

    res = 1;
    for (i = 0; i < n * n * 3; ++i)
    {
        if (abs(greyImage[i] - imImage[i]) > 1)
        {
            printf("\nAt: %u\nIs: %u\nShould be: %u\n", i, imImage[i], greyImage[i]);
            res = 0;
            break;
        }
    }
    // Free all resources...
    free(image);
    for (i = 0; i < n; ++i)
        free(cpxImg[i]);
    free(cpxImg);
    free(imImage);
    free(imImage2);
    free(greyImage);
    return res;
}

int run_fbtest(fft_function fn, const int n)
{
    int i;
    tb_cpx *in = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    tb_cpx *out = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    float *reference = (float *)malloc(sizeof(float)*n);
    for (i = 0; i < n; ++i)
    {
        reference[i] = 0;//sin(M_2_PI * ((float)i / n));
        in[i].r = reference[i];
        in[i].i = 0.f;
    }
    in[1].r = reference[1] = 1.f;
    fn(FORWARD_FFT, in, out, n);
    fn(INVERSE_FFT, out, in, n);
    for (i = 0; i < n; ++i)
    {
        if (abs(reference[i] - in[i].r) > ERROR_MARGIN)
        {
            free(in);
            free(out);
            free(reference);
            return 0;
        }
    }
    free(in);
    free(out);
    free(reference);
    return 1;
}

int run_fft2dinvtest(fft_function fn, const int n)
{
    int i, x, y;
    tb_cpx **seq2d, **ref;
    char *format = "{%.2f, %.2f} ";
    seq2d = (tb_cpx **)malloc(sizeof(tb_cpx) * n * n);
    ref = (tb_cpx **)malloc(sizeof(tb_cpx) * n * n);
    for (i = 0; i < n; ++i)
    {
        seq2d[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
        ref[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    }
    for (y = 0; y < n; ++y)
    {
        for (x = 0; x < n; ++x)
        {
            seq2d[y][x].r = ref[y][x].r = 0.f;
            seq2d[y][x].i = ref[y][x].i = 0.f;
        }
    }
    seq2d[1][1].r = ref[1][1].r = 1.f;
    tb_fft2d(FORWARD_FFT, fn, seq2d, n);
    tb_fft2d(INVERSE_FFT, fn, seq2d, n);
    for (y = 0; y < n; ++y)
    {
        for (x = 0; x < n; ++x)
        {
            if (abs(seq2d[y][x].r - ref[y][x].r) > ERROR_MARGIN || abs(seq2d[y][x].i - ref[y][x].i) > ERROR_MARGIN)
            {
                for (i = 0; i < n; ++i)
                {
                    free(seq2d);
                    free(ref);
                }
                free(seq2d);
                free(ref);
                return 0;
            }
        }
    }
    for (i = 0; i < n; ++i)
    {
        free(seq2d);
        free(ref);
    }
    free(seq2d);
    free(ref);
    return 1;
}

int run_fft2dTest(fft_function fn, const int n)
{
    int res, w, m;
    int i;
    char filename[13];
    unsigned char *image, *imImage;
    tb_cpx **cpxImg;

    sprintf_s(filename, 13, "lena_%u.ppm", n);
    /* Read image to memory */
    image = readppm(filename, &w, &m);
    if (n != n || m != n)
    {
        printf("Image size not square and pw of 2.\n");
        getchar();
        return 0;
    }
    /* Allocate resources */
    imImage = (unsigned char *)malloc(sizeof(unsigned char) * n * n * 3);
    cpxImg = (tb_cpx **)malloc(sizeof(tb_cpx *) * n);
    for (i = 0; i < n; ++i)
    {
        cpxImg[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    }

    /* Set real values from image values.
    * Store the real-value version of the image.
    */
    img_to_cpx(image, cpxImg, n);
    cpx_to_img(cpxImg, image, n, 0);
    tb_fft2d(FORWARD_FFT, tb_fft, cpxImg, n);
    tb_fft2d(INVERSE_FFT, tb_fft, cpxImg, n);
    cpx_to_img(cpxImg, imImage, n, 0);
    writeppm("img-out.ppm", n, n, imImage);

    res = 1;
    for (i = 0; i < n * n * 3; ++i)
    {
        if (abs(image[i] - imImage[i]) > 1)
        {
            res = 0;
            break;
        }
    }
    // Free all resources...
    free(image);
    for (i = 0; i < n; ++i)
        free(cpxImg[i]);
    free(cpxImg);
    free(imImage);
    return res;
}

void prntTrans(tb_cpx **seq, const int n)
{
    int x, y, w, h;
    w = n < 11 ? n : 10;
    h = n < 11 ? n : 10;
    printf("\n");
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            printf("(%.0f, %.0f) ", seq[y][x].r, seq[y][x].i);
        }
        printf("\n");
    }
}

unsigned char test_transpose(transpose_function fn, const int b, const int n)
{
    int x, y;
    tb_cpx **in;
    in = (tb_cpx **)malloc(sizeof(tb_cpx *) * n);
    for (y = 0; y < n; ++y) {
        in[y] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
        for (x = 0; x < n; ++x) {
            in[y][x] = { (float)x, (float)y };
        }
    }
    fn(in, b, n);
    for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
            if (in[y][x].r != (float)y || in[x][y].r != (float)x) {
                printf("Error no block at (%u, %u) is: (%f,%f)\n", x, y, in[x][y].r, in[x][y].i);
                return 0;
            }
        }
    }
    return 1;
}

double test_time_transpose(transpose_function trans_fn, const int b, const int n)
{
    int x, y;
    LARGE_INTEGER freq, tStart, tStop;
    double measures[NO_TESTS];
    tb_cpx **in = (tb_cpx **)malloc(sizeof(tb_cpx *) * n);
    for (y = 0; y < n; ++y) {
        in[y] = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
        for (x = 0; x < n; ++x) {
            in[y][x] = { (float)x, (float)y };
        }
    }
    trans_fn(in, b, n);
    trans_fn(in, b, n);
    QueryPerformanceFrequency(&freq);
    for (int i = 0; i < NO_TESTS; ++i) {
        QueryPerformanceCounter(&tStart);
        trans_fn(in, b, n);
        QueryPerformanceCounter(&tStop);
        measures[i] = (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }

    for (y = 0; y < n; ++y)
        free(in[y]);
    free(in);
    return avg(measures, NO_TESTS);
}

void kiss_fft(double dir, tb_cpx *in, tb_cpx *out, const int n)
{
    kiss_fft_cfg cfg = kiss_fft_alloc(n, (dir == INVERSE_FFT), NULL, NULL);
    kiss_fft(cfg, in, out);
    free(cfg);
}