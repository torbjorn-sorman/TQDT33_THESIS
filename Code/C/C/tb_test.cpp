#include <Windows.h>
#include <limits>

#include "tb_test.h"
#include "tb_fft.h"
#include "tb_image.h"

#define NO_TESTS 64
#define ERROR_MARGIN 0.0002

int checkError(tb_cpx *seq, tb_cpx *ref, uint32_t N, int print)
{
    uint32_t n;
    double r, i, i_val, r_val;
    r = i = 0.0;
    for (n = 0; n < N; ++n)
    {
        r_val = abs(seq[n].r - ref[n].r);
        i_val = abs(seq[n].i - ref[n].i);
        r = r > r_val ? r : r_val;
        i = i > i_val ? i : i_val;
    }
    if (print == 1) printf("Error e, abs(real(e)): %f\t abs(imag(e)): %f\n", r, i);
    return r > ERROR_MARGIN || i > ERROR_MARGIN;
}

unsigned char test_equal_dft(fft_function fft_fn, fft_function dft_ref_fn, uint32_t N, uint32_t inplace)
{
    uint32_t i;
    unsigned char res = 1;
    tb_cpx *fft_out, *dft_out, *in, *in2;

    in = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    for (i = 0; i < N; ++i)
    {
        in[i].r = (float)sin(M_2_PI * (((double)i) / N));
        in[i].i = 0.f;
    }
    if (inplace == 0)
    {
        fft_out = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        dft_out = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        dft_ref_fn(FORWARD_FFT, in, dft_out, N);
        fft_fn(FORWARD_FFT, in, fft_out, N);
        res = checkError(dft_out, fft_out, N, 1);
        free(dft_out);
        free(fft_out);
    }
    else
    {
        in2 = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        for (i = 0; i < N; ++i) {
            in2[i].r = in[i].r;
            in2[i].i = in[i].i;
        }
        dft_ref_fn(FORWARD_FFT, in, in, N);
        fft_fn(FORWARD_FFT, in2, in2, N);
        res = checkError(in, in2, N, 1);
        free(in2);
    }
    free(in);
    return res;
}

double test_time_dft(fft_function fft_fn, uint32_t N)
{
    LARGE_INTEGER freq, tStart, tStop;
    tb_cpx *in = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    tb_cpx *out = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    double m = DBL_MAX;
    uint32_t i;
    for (i = 0; i < N; ++i) {
        in[i].r = (float)sin(M_2_PI * ((double)i / N));
        in[i].i = 0.f;
    }
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&tStart);
    for (int i = 0; i < (NO_TESTS * 4 / log2_32(N)); ++i) {
        fft_fn(FORWARD_FFT, in, out, N);
    }
    QueryPerformanceCounter(&tStop);
    m = min(m, (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart);
    free(in);
    free(out);
    return m / (NO_TESTS * 4);
}

double test_time_dft_2d(fft_function fft_fn, uint32_t N)
{
    int x, y;
    LARGE_INTEGER freq, tStart, tStop;
    double m, tmp;
    char filename[30];
    unsigned char *image;
    tb_cpx **cxImage;
    uint32_t i;

    cxImage = (tb_cpx **)malloc(sizeof(tb_cpx) * N);
    for (i = 0; i < N; ++i)
    {
        cxImage[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    }    
    sprintf_s(filename, 30, "photo05_%u.ppm", N);
    image = readppm(filename, &x, &y);        
    m = DBL_MAX;
    QueryPerformanceFrequency(&freq);
    for (int i = 0; i < NO_TESTS / 4; ++i)
    {
        img_to_cpx(image, cxImage, N);
        QueryPerformanceCounter(&tStart);
        tb_fft2d(FORWARD_FFT, fft_fn, cxImage, N);
        QueryPerformanceCounter(&tStop);
        tmp = (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart;
        m = tmp < m ? tmp : m;
    }
    free(image);
    for (i = 0; i < N; ++i)
        free(cxImage[i]);
    free(cxImage);
    return (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart;
}

unsigned char test_image(fft_function fft_fn, char *filename, uint32_t N)
{
    int res, n, m;
    uint32_t i;
    char file[30];
    unsigned char *image, *imImage, *imImage2, *greyImage;
    tb_cpx **cxImage;
    sprintf_s(file, 30, "%s_%u.ppm", filename, N);
    printf("Read: %s\n", file);
    image = readppm(file, &n, &m);
    if (n != N || m != N)
    {
        printf("Image size not square and pw of 2.\n");
        getchar();
        return 0;
    }
    greyImage = (unsigned char *)malloc(sizeof(unsigned char) * N * N * 3);
    imImage = (unsigned char *)malloc(sizeof(unsigned char) * N * N * 3);
    imImage2 = (unsigned char *)malloc(sizeof(unsigned char) * N * N * 3);
    cxImage = (tb_cpx **)malloc(sizeof(tb_cpx) * N);
    for (i = 0; i < N; ++i)
    {
        cxImage[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    }

    /* Set real values from image values.
    * Store the real-value version of the image.
    */
    img_to_cpx(image, cxImage, N);
    cpx_to_img(cxImage, greyImage, N, 0);
    printf("Write img00-grey.ppm\n");
    writeppm("img00-grey.ppm", N, N, greyImage);

    /* Run 2D FFT on complex values.
    * Map absolute values of complex to pixels and store to file.
    */
    tb_fft2d(FORWARD_FFT, tb_fft, cxImage, N);
    cpx_to_img(cxImage, imImage, N, 1);
    fft_shift(imImage, imImage2, N);
    printf("Write img01-magnitude.ppm\n");
    writeppm("img01-magnitude.ppm", N, N, imImage2);

    /* Run inverse 2D FFT on complex values */
    tb_fft2d(INVERSE_FFT, tb_fft, cxImage, N);
    cpx_to_img(cxImage, imImage, N, 0);
    printf("Write img02-fftToImage.ppm\n");
    writeppm("img02-fftToImage.ppm", N, N, imImage);

    res = 1;
    for (i = 0; i < N * N * 3; ++i)
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
    for (i = 0; i < N; ++i)
        free(cxImage[i]);
    free(cxImage);
    free(imImage);
    free(imImage2);
    free(greyImage);
    return res;
}

int run_fbtest(fft_function fn, uint32_t N)
{
    uint32_t i;
    tb_cpx *in = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    tb_cpx *out = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    float *reference = (float *)malloc(sizeof(float)*N);
    for (i = 0; i < N; ++i)
    {
        reference[i] = 0;//sin(M_2_PI * ((float)i / N));
        in[i].r = reference[i];
        in[i].i = 0.f;
    }
    in[1].r = reference[1] = 1.f;
    fn(FORWARD_FFT, in, out, N);
    fn(INVERSE_FFT, out, in, N);
    for (i = 0; i < N; ++i)
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

int run_fft2dinvtest(fft_function fn, uint32_t N)
{
    uint32_t i, x, y;
    tb_cpx **seq2d, **ref;
    char *format = "{%.2f, %.2f} ";
    seq2d = (tb_cpx **)malloc(sizeof(tb_cpx) * N * N);
    ref = (tb_cpx **)malloc(sizeof(tb_cpx) * N * N);
    for (i = 0; i < N; ++i)
    {
        seq2d[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        ref[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    }
    for (y = 0; y < N; ++y)
    {
        for (x = 0; x < N; ++x)
        {
            seq2d[y][x].r = ref[y][x].r = 0.f;
            seq2d[y][x].i = ref[y][x].i = 0.f;
        }
    }
    seq2d[1][1].r = ref[1][1].r = 1.f;
    tb_fft2d(FORWARD_FFT, fn, seq2d, N);
    tb_fft2d(INVERSE_FFT, fn, seq2d, N);
    for (y = 0; y < N; ++y)
    {
        for (x = 0; x < N; ++x)
        {
            if (abs(seq2d[y][x].r - ref[y][x].r) > ERROR_MARGIN || abs(seq2d[y][x].i - ref[y][x].i) > ERROR_MARGIN)
            {
                for (i = 0; i < N; ++i)
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
    for (i = 0; i < N; ++i)
    {
        free(seq2d);
        free(ref);
    }
    free(seq2d);
    free(ref);
    return 1;
}



int run_fft2dTest(fft_function fn, uint32_t N)
{
    int res, n, m;
    uint32_t i;
    char filename[13];
    unsigned char *image, *imImage;
    tb_cpx **cxImage;

    sprintf_s(filename, 13, "lena_%u.ppm", N);
    /* Read image to memory */
    image = readppm(filename, &n, &m);
    if (n != N || m != N)
    {
        printf("Image size not square and pw of 2.\n");
        getchar();
        return 0;
    }
    /* Allocate resources */
    imImage = (unsigned char *)malloc(sizeof(unsigned char) * N * N * 3);
    cxImage = (tb_cpx **)malloc(sizeof(tb_cpx) * N);
    for (i = 0; i < N; ++i)
    {
        cxImage[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    }

    /* Set real values from image values.
    * Store the real-value version of the image.
    */
    img_to_cpx(image, cxImage, N);
    cpx_to_img(cxImage, image, N, 0);
    tb_fft2d(FORWARD_FFT, tb_fft, cxImage, N);
    tb_fft2d(INVERSE_FFT, tb_fft, cxImage, N);
    cpx_to_img(cxImage, imImage, N, 0);
    writeppm("img-out.ppm", N, N, imImage);

    res = 1;
    for (i = 0; i < N * N * 3; ++i)
    {
        if (abs(image[i] - imImage[i]) > 1)
        {
            res = 0;
            break;
        }
    }
    // Free all resources...
    free(image);
    for (i = 0; i < N; ++i)
        free(cxImage[i]);
    free(cxImage);
    free(imImage);
    return res;
}

void kiss_fft(int dir, tb_cpx *in, tb_cpx *out, uint32_t N)
{
    kiss_fft_cfg cfg = kiss_fft_alloc(N, (dir == INVERSE_FFT), NULL, NULL);
    kiss_fft(cfg, in, out);
    free(cfg);
}