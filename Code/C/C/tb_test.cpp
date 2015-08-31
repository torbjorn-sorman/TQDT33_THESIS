#include <Windows.h>
#include <limits>

#include "tb_test.h"
#include "tb_test_helper.h"
#include "tb_fft.h"
#include "tb_fft_helper.h"
#include "tb_image.h"
#include "tb_transpose.h"
#include "tb_math.h"
#include "tb_print.h"

#define NO_TESTS 8
#define MAX_LENGTH 1048576

typedef LARGE_INTEGER LI;
#define QPF QueryPerformanceFrequency
#define QPC QueryPerformanceCounter

#define MEASURE_TIME(RES, FN) LI s, e, em, f; QPF(&f); QPC(&s); FN; QPC(&e); em.QuadPart = e.QuadPart - s.QuadPart; em.QuadPart *= 1000000; em.QuadPart /= f.QuadPart; RES = (double)em.QuadPart;

double measureTime()
{
    LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
    LARGE_INTEGER Frequency;
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);
    // Activity to measure!
    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    return (double)ElapsedMicroseconds.QuadPart;
}

void simple()
{
    const int n = 16;
    tb_cpx *in, *in2, *out, *W;
    W = NULL;
    out = get_seq(n);
    in = get_seq(n);
    in[1].r = 1.f;
    in2 = get_seq(n, in);

    console_print(in, n);
    console_separator(1);
    tb_fft(FORWARD_FFT, in, in, W, n);
    tb_fft_alt(FORWARD_FFT, in2, in2, W, n);
    console_print_cmp(in, in2, n);

    console_separator(1);    
    free(W);
    free(in);
    free(in2);
    free(out);
}

unsigned char test_equal_dft(fft_function fft_fn, fft_function dft_ref_fn, const int inplace)
{
    int n;
    unsigned char res = 1;
    tb_cpx *fft_out, *dft_out, *in, *in2, *W;

    for (n = 2; n < MAX_LENGTH; n *= 2) {
        W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
        twiddle_factors(W, 32 - log2_32(n), n);
        in = get_seq(n, 1);
        if (inplace == 0)
        {            
            fft_out = get_seq(n);
            dft_out = get_seq(n);
            dft_ref_fn(FORWARD_FFT, in, dft_out, W, n);
            fft_fn(FORWARD_FFT, in, fft_out, W, n);
            res = checkError(dft_out, fft_out, n, 1);
            free(dft_out);
            free(fft_out);
        }
        else
        {
            in2 = get_seq(n, in);
            dft_ref_fn(FORWARD_FFT, in, in, W, n);
            fft_fn(FORWARD_FFT, in2, in2, W, n);
            res = checkError(in, in2, n, 1);
            free(W);
            free(in2);
        }
        free(W);
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
    double measures[NO_TESTS];
    tb_cpx *in, *out, *W;
    

    in = get_seq(n, 1);
    out = get_seq(n);
    for (i = 0; i < NO_TESTS; ++i) {
        MEASURE_TIME(measures[i], 
            W = (tb_cpx *)malloc(sizeof(tb_cpx) * n); 
            twiddle_factors(W, 32 - log2_32(n), n); 
            fft_fn(FORWARD_FFT, in, out, W, n); 
            free(W)
        );
    }
    free(in);
    free(out);
    return avg(measures, NO_TESTS);
}

double test_time_dft_2d(fft2d_function fft2d, fft_function fft_fn, const int n)
{
    int i, x, y;
    char filename[30];
    unsigned char *image;
    tb_cpx **cpxImg, **cpxImgRef;
    double measures[NO_TESTS];

    cpxImg = get_seq2d(n);
    cpxImgRef = get_seq2d(n);
    sprintf_s(filename, 30, "img/lena/%u.ppm", n);
    image = readppm(filename, &x, &y);
    img_to_cpx(image, cpxImgRef, n);

    for (i = 0; i < NO_TESTS; ++i)
    {
        copy_seq2d(cpxImgRef, cpxImg, n);
        MEASURE_TIME(measures[i], fft2d(FORWARD_FFT, fft_fn, cpxImg, n); fft2d(INVERSE_FFT, fft_fn, cpxImg, n));
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
        printf("(ms)\t%.2f\t%.1f\t%.1f\t%.1f\t%u\n", diff / time_ref, diff, time, time_ref, n);

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
    fft2d(FORWARD_FFT, fft_fn, cpxImg, n);
    cpx_to_img(cpxImg, imImage, n, 1);
    fft_shift(imImage, imImage2, n);
    printf("Write img01-magnitude.ppm\n");
    writeppm("img01-magnitude.ppm", n, n, imImage2);

    /* Run inverse 2D FFT on complex values */
    fft2d(INVERSE_FFT, fft_fn, cpxImg, n);
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
    double measures[NO_TESTS];
    tb_cpx **in;
    in = get_seq2d(n, 2);
    for (i = 0; i < NO_TESTS; ++i) {
        MEASURE_TIME(measures[i], 
            trans_fn(in, b, n)
        );
    }
    free_seq2d(in, n);
    return avg(measures, NO_TESTS);
}

double test_time_twiddle(twiddle_function fn, const int n)
{
    int i, lead;
    double measures[NO_TESTS];
    tb_cpx *w;
    lead = 32 - log2_32(n);
    w = get_seq(n);    
    for (i = 0; i < NO_TESTS; ++i) {
        MEASURE_TIME(measures[i], 
            fn(w, lead, n)
        );
    }
    free(w);
    return avg(measures, NO_TESTS);
}

unsigned char test_twiddle(twiddle_function fn, twiddle_function ref, const int n)
{
    int lead, i, res;
    tb_cpx *w, *w_ref;
    lead = 32 - log2_32(n);
    w = get_seq(n);
    w_ref = get_seq(n);

    fn(w, lead, n);
    ref(w_ref, lead, n);

    res = 1;
    for (i = 0; i < n; ++i) {
        if (abs_diff(w[i], w_ref[i]) > 0.00001) {
            printf("Diff: %f\n", abs_diff(w[i], w_ref[i]));
            res = 0;
            break;
        }
    }
    free(w);
    free(w_ref);
    return res;
}

tb_cpx *makeRefTW(double dir, int n)
{
    int i;
    tb_cpx *ref;
    ref = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    double ang = dir * (M_2_PI / n);
    for (i = 0; i < n; ++i) {
        ref[i].r = cos(ang * i);
        ref[i].i = sin(ang * i);
    }
    bit_reverse(ref, FORWARD_FFT, n, 32 - log2_32(n));
    return ref;
}

void test_twiddle_delux()
{
    int n;
    tb_cpx *W, *ref;
    n = 16;
    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    twiddle_factors(W, 32 - log2_32(n), n);
    console_print(W, n);
    console_separator(1);
    console_print(ref = makeRefTW(-1.0, n), n);
    free(ref);
    console_separator(3);
    twiddle_factors_inverse(W, n);
    console_print(W, n);
    console_separator(1);
    console_print(ref = makeRefTW(1.0, n), n);
    free(ref);
    console_separator(1);
}

void test_complete_fft(char *name, fft_function fn)
{
    int n, i;
    tb_cpx *in, *ref, *W;
    n = 16;
    in = get_seq(n);
    in[1].r = 1;
    ref = get_seq(n, in);
    printf("\n%s\n", name);

    W = (tb_cpx *)malloc(sizeof(tb_cpx) * n);
    twiddle_factors(W, 32 - log2_32(n), n);
    fn(FORWARD_FFT, in, in, W, n);

    twiddle_factors_inverse(W, n);
    console_print(in, n);
    console_separator(1);  
    fn(INVERSE_FFT, in, in, W, n);
    free(W);

    console_print(ref, n);
    console_separator(1);
    checkError(in, ref, n, 1);
    console_separator(1);
    console_print(in, n);
    console_separator(1);

    /*
    printf("Length\tTime\n");
    for (i = 4; i < MAX_LENGTH; i *= 2) {
        printf("%d\t%.1f\n", i, test_time_dft(fn, i));
    }
    */

    free(in);
    free(ref);
}

void test_complete_fft2d(char *name, fft2d_function fn)
{
    int n, i;
    tb_cpx **in, **ref;
    fft_function fft;
    fft = tb_fft;
    n = 512;
    in = get_seq2d(n, 1);    
    ref = get_seq2d(n, in);

    fn(FORWARD_FFT, fft, in, n);
    fn(INVERSE_FFT, fft, in, n);
    printf("\n%s\n", name);
    checkError(in, ref, n, 1);

    printf("Length\tTime\n");
    for (i = 4; i < 4096; i *= 2) {
        printf("%d\t%.1f\n", i, test_time_dft_2d(fn, fft, i));
    }

    free(in);
    free(ref);
}

void kiss_fft(double dir, tb_cpx *in, tb_cpx *out, const int n)
{
    kiss_fft_cfg cfg = kiss_fft_alloc(n, (dir == INVERSE_FFT), NULL, NULL);
    kiss_fft(cfg, in, out);
    free(cfg);
}