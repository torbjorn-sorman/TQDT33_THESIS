#include <Windows.h>
#include <limits>

#include "tb_test.h"
#include "tb_test_helper.h"
#include "tb_fft.h"
#include "tb_image.h"
#include "tb_transpose.h"
#include "tb_math.h"
#include "tb_print.h"

#define NO_TESTS 64
#define MAX_LENGTH 1048576

void simple()
{
    const int n = 16;
    tb_cpx *in, *in2, *out;
    in = get_seq(n);
    in2 = get_seq(n);
    out = get_seq(n);
    in[1].r = 1.f;
    in2[1] = in[1];
    console_print(in, n);
    console_separator(1);
    tb_fft(FORWARD_FFT, in2, in2, n);
    console_print(out, n);
    console_separator(1);
    console_print(in2, n);
    free(in);
    free(in2);
    free(out);
}

unsigned char test_equal_dft(fft_function fft_fn, fft_function dft_ref_fn, const int inplace)
{
    int n;
    unsigned char res = 1;
    tb_cpx *fft_out, *dft_out, *in, *in2;

    for (n = 2; n < MAX_LENGTH; n *= 2) {
        in = get_seq(n, 1);
        if (inplace == 0)
        {            
            fft_out = get_seq(n);
            dft_out = get_seq(n);
            dft_ref_fn(FORWARD_FFT, in, dft_out, n);
            fft_fn(FORWARD_FFT, in, fft_out, n);
            res = checkError(dft_out, fft_out, n, 1);
            free(dft_out);
            free(fft_out);
        }
        else
        {
            in2 = get_seq(n, in);
            dft_ref_fn(FORWARD_FFT, in, in, n);
            fft_fn(FORWARD_FFT, in2, in2, n);
            res = checkError(in, in2, n, 1);
            free(in2);
        }
        free(in);
    }

    return res;
}

unsigned char test_equal_dft2d(fft2d_function fft_2d, fft_function fft_fn, fft_function ref_fn, const int inplace)
{
    int n, m, size;
    unsigned char *image, *imImage, *imImageRef;
    char filename[30];
    tb_cpx **cpxImg, **cpxImgRef;
    size = 4096;
    sprintf_s(filename, 30, "img/lena/%u.ppm", size);
    image = readppm(filename, &n, &m);
    imImage = get_empty_img(size, size);
    imImageRef = get_empty_img(size, size);
    cpxImg = get_seq2d(n);
    cpxImgRef = get_seq2d(n);
    img_to_cpx(image, cpxImg, size);
    img_to_cpx(image, cpxImgRef, size);

    fft_2d(FORWARD_FFT, fft_fn, cpxImg, size);
    fft_2d(FORWARD_FFT, ref_fn, cpxImgRef, size);

    printf("Max Fw error: %f\n", cpx_diff(cpxImg, cpxImgRef, size));
    printf("Avg Fw error: %f\n", cpx_avg_diff(cpxImg, cpxImgRef, size));

    fft_2d(INVERSE_FFT, fft_fn, cpxImg, size);
    fft_2d(INVERSE_FFT, ref_fn, cpxImgRef, size);

    printf("Max Inv error: %f\n", cpx_diff(cpxImg, cpxImgRef, size));
    printf("Avg Inv error: %f\n", cpx_avg_diff(cpxImg, cpxImgRef, size));

    cpx_to_img(cpxImg, imImage, size, 0);
    cpx_to_img(cpxImgRef, imImageRef, size, 0);

    writeppm("test_equal.ppm", size, size, imImage);
    writeppm("test_equal_ref.ppm", size, size, imImageRef);

    free(image);
    free_seq2d(cpxImg, n);
    free_seq2d(cpxImgRef, n);
    free(imImage);
    free(imImageRef);
    return 1;
}

double test_time_dft(fft_function fft_fn, const int n)
{
    int i;    
    LARGE_INTEGER freq, tStart, tStop;
    double measures[NO_TESTS];
    tb_cpx *in, *out;

    in = get_seq(n, 1);
    out = get_seq(n);
    QueryPerformanceFrequency(&freq);
    for (i = 0; i < NO_TESTS; ++i) {
        QueryPerformanceCounter(&tStart);
        fft_fn(FORWARD_FFT, in, out, n);
        QueryPerformanceCounter(&tStop);
        measures[i] = (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart;
    }
    free(in);
    free(out);
    return avg(measures, NO_TESTS);
}

double test_time_dft_2d(fft2d_function fft2d, fft_function fft_fn, const int n)
{
    int i, x, y;
    LARGE_INTEGER freq, tStart, tStop;
    char filename[30];
    unsigned char *image;
    tb_cpx **cpxImg, **cpxImgRef;
    double measures[NO_TESTS];

    cpxImg = get_seq2d(n);
    cpxImgRef = get_seq2d(n);
    sprintf_s(filename, 30, "img/lena/%u.ppm", n);
    image = readppm(filename, &x, &y);
    img_to_cpx(image, cpxImgRef, n);

    QueryPerformanceFrequency(&freq);
    for (i = 0; i < NO_TESTS; ++i)
    {
        copy_seq2d(cpxImgRef, cpxImg, n);
        QueryPerformanceCounter(&tStart);
        fft2d(FORWARD_FFT, fft_fn, cpxImg, n);
        fft2d(INVERSE_FFT, fft_fn, cpxImg, n);
        QueryPerformanceCounter(&tStop);
        measures[i] = (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart;
    }
    free(image);
    free_seq2d(cpxImg, n);
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
    for (n = 8; n < MAX_LENGTH; n *= 2) {
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

unsigned char test_image(fft2d_function fft2d, fft_function fft_fn, char *filename, const int n)
{
    int res, w, m, i;
    char file[30];
    unsigned char *image, *imImage, *imImage2, *greyImage;
    tb_cpx **cpxImg;

    sprintf_s(file, 30, "img/%s/%u.ppm", filename, n);
    printf("Read: %s\n", file);
    image = readppm(file, &w, &m);

    if (w != n || m != n)
        return 0;

    greyImage = get_empty_img(n, n);
    imImage = get_empty_img(n, n);
    imImage2 = get_empty_img(n, n);
    cpxImg = get_seq2d(n);

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
    fft2d(FORWARD_FFT, tb_fft, cpxImg, n);
    cpx_to_img(cpxImg, imImage, n, 1);
    fft_shift(imImage, imImage2, n);
    printf("Write img01-magnitude.ppm\n");
    writeppm("img01-magnitude.ppm", n, n, imImage2);

    /* Run inverse 2D FFT on complex values */
    fft2d(INVERSE_FFT, tb_fft, cpxImg, n);
    cpx_to_img(cpxImg, imImage, n, 0);
    printf("Write img02-fftToImage.ppm\n");
    writeppm("img02-fftToImage.ppm", n, n, imImage);

    res = cmp(imImage, greyImage);
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

unsigned char test_transpose(transpose_function fn, const int b, const int n)
{
    int x, y, res;
    tb_cpx **in;
    in = get_seq2d(n, 2);
    fn(in, b, n);
    res = 1;
    for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
            if (in[y][x].r != (float)y || in[x][y].r != (float)x) {
                printf("Error no block at (%u, %u) is: (%f,%f)\n", x, y, in[x][y].r, in[x][y].i);
                res = 0;
                break;
            }
        }
    }
    free_seq2d(in, n);
    return res;
}

double test_time_transpose(transpose_function trans_fn, const int b, const int n)
{
    int i;
    LARGE_INTEGER freq, tStart, tStop;
    double measures[NO_TESTS];
    tb_cpx **in;
    in = get_seq2d(n, 2);
    trans_fn(in, b, n);
    trans_fn(in, b, n);
    QueryPerformanceFrequency(&freq);
    for (i = 0; i < NO_TESTS; ++i) {
        QueryPerformanceCounter(&tStart);
        trans_fn(in, b, n);
        QueryPerformanceCounter(&tStop);
        measures[i] = (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }
    free_seq2d(in, n);
    return avg(measures, NO_TESTS);
}

double test_time_twiddle(twiddle_function fn, const int n)
{
    int i, lead;
    LARGE_INTEGER freq, tStart, tStop;
    double measures[NO_TESTS];
    tb_cpx *w;
    lead = 32 - log2_32(n);
    w = get_seq(n);    
    QueryPerformanceFrequency(&freq);
    for (i = 0; i < NO_TESTS; ++i) {
        QueryPerformanceCounter(&tStart);
        fn(w, FORWARD_FFT, lead, n);
        QueryPerformanceCounter(&tStop);
        measures[i] = (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (double)freq.QuadPart;
    }
    free(w);
    return avg(measures, NO_TESTS);
}

unsigned char test_twiddle(twiddle_function fn, twiddle_function ref, const int n)
{
    int lead, i;
    tb_cpx *w, *w_ref;
    lead = 32 - log2_32(n);
    w = get_seq(n);
    w_ref = get_seq(n);

    fn(w, FORWARD_FFT, lead, n);
    ref(w_ref, FORWARD_FFT, lead, n);

    for (i = 0; i < n; ++i) {
        if (abs_diff(w[i], w_ref[i]) > 0.00001) {
            printf("Diff: %f\n", abs_diff(w[i], w_ref[i]));
            return 0;
        }
    }
    return 1;
}

void kiss_fft(double dir, tb_cpx *in, tb_cpx *out, const int n)
{
    kiss_fft_cfg cfg = kiss_fft_alloc(n, (dir == INVERSE_FFT), NULL, NULL);
    kiss_fft(cfg, in, out);
    free(cfg);
}