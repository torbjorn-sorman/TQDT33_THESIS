#include <Windows.h>
#include <limits>

#include "tb_test.h"
#include "tb_fft.h"
#include "tb_image.h"
#include "tb_transpose.h"
#include "tb_math.h"

#define NO_TESTS 16
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
    if (print == 1) printf("Error %u\tre(e): %f\t im(e): %f\n", N, r, i);
    return r > ERROR_MARGIN || i > ERROR_MARGIN;
}

unsigned char test_equal_dft(fft_function fft_fn, fft_function dft_ref_fn, uint32_t inplace)
{
    uint32_t i, N;
    unsigned char res = 1;
    tb_cpx *fft_out, *dft_out, *in, *in2;

    for (N = 2; N < 4194304; N *= 2) {
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
    }
    return res;
}

unsigned char test_equal_dft2d(fft_function fft_fn, fft_function ref_fn, uint32_t inplace)
{
    int n, m;
    uint32_t i, N;    
    unsigned char *image, *imImage, *imImageRef;
    tb_cpx **cpxImg, **cpxImgRef;

    N = 512;
    image = readppm("lena_512.ppm", &n, &m);
    imImage = (unsigned char *)malloc(sizeof(unsigned char) * N * N * 3);
    imImageRef = (unsigned char *)malloc(sizeof(unsigned char) * N * N * 3);
    cpxImg = (tb_cpx **)malloc(sizeof(tb_cpx) * N);
    cpxImgRef = (tb_cpx **)malloc(sizeof(tb_cpx) * N);
    for (i = 0; i < N; ++i) {
        cpxImg[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        cpxImgRef[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    }
    img_to_cpx(image, cpxImg, N);
    img_to_cpx(image, cpxImgRef, N);

    tb_fft2d_inplace(FORWARD_FFT, fft_fn, cpxImg, N);
    tb_fft2d_inplace(FORWARD_FFT, ref_fn, cpxImgRef, N);    

    printf("Max error: %f\n", cpx_diff(cpxImg, cpxImgRef, N));
    printf("Avg error: %f\n", cpx_avg_diff(cpxImg, cpxImgRef, N));

    tb_fft2d_inplace(INVERSE_FFT, fft_fn, cpxImg, N);
    tb_fft2d_inplace(INVERSE_FFT, ref_fn, cpxImgRef, N);
        
    cpx_to_img(cpxImg, imImage, N, 0);
    cpx_to_img(cpxImgRef, imImageRef, N, 0);

    writeppm("test_equal.ppm", N, N, imImage);
    writeppm("test_equal_ref.ppm", N, N, imImageRef);
        
    free(image);
    for (i = 0; i < N; ++i) {
        free(cpxImg[i]);
        free(cpxImgRef[i]);
    }
    free(cpxImg);
    free(cpxImgRef);
    free(imImage);
    free(imImageRef);
    return 1;
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
    char filename[30];
    unsigned char *image;
    tb_cpx **cpxImg, **cpxImgRef;
    uint32_t i, k, l;

    cpxImg = (tb_cpx **)malloc(sizeof(tb_cpx) * N);
    cpxImgRef = (tb_cpx **)malloc(sizeof(tb_cpx) * N);
    for (i = 0; i < N; ++i) {
        cpxImg[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        cpxImgRef[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    }    
    sprintf_s(filename, 30, "lena_%u.ppm", N);
    image = readppm(filename, &x, &y);
    img_to_cpx(image, cpxImgRef, N);

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&tStart);
    for (i = 0; i < NO_TESTS; ++i)
    {
        for (k = 0; k < N; ++k) {
            for (l = 0; l < N; ++l) {
                cpxImg[k][l].r = cpxImgRef[k][l].r;
                cpxImg[k][l].i = cpxImgRef[k][l].i;
            }
        }
        tb_fft2d_inplace(FORWARD_FFT, fft_fn, cpxImg, N);
        tb_fft2d_inplace(INVERSE_FFT, fft_fn, cpxImg, N);
    }
    QueryPerformanceCounter(&tStop);
    free(image);
    for (i = 0; i < N; ++i)
        free(cpxImg[i]);
    free(cpxImg);
    return ((double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart) / NO_TESTS;
}

double test_cmp_time(fft_function fn, fft_function ref)
{
    uint32_t n;
    double time, time_ref, diff, rel, sum, sum_ref;
    rel = DBL_MIN;
    sum = sum_ref = 0.0;
    printf("     rel.\tdiff.\t\ttime\t\tref\t\tN\n");
    for (n = 4; n < 16777216; n *= 2) {
        time = test_time_dft(fn, n);
        time_ref = test_time_dft(ref, n);        
        diff = time_ref - time; 
        sum += time;
        sum_ref += time_ref;
        rel += diff / time_ref;
        printf("(ms) %f\t%f\t%f\t%f\t%u\n", diff / time_ref, diff, time, time_ref, n);
        
    }
    return rel / 22;
}

unsigned char test_image(fft_function fft_fn, char *filename, uint32_t N)
{
    int res, n, m;
    uint32_t i;
    char file[30];
    unsigned char *image, *imImage, *imImage2, *greyImage;
    tb_cpx **cpxImg;
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
    cpxImg = (tb_cpx **)malloc(sizeof(tb_cpx) * N);
    for (i = 0; i < N; ++i)
    {
        cpxImg[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    }

    /* Set real values from image values.
    * Store the real-value version of the image.
    */
    img_to_cpx(image, cpxImg, N);
    writeppm("img00-org.ppm", N, N, image);
    cpx_to_img(cpxImg, greyImage, N, 0);
    printf("Write img00-grey.ppm\n");
    writeppm("img00-grey.ppm", N, N, greyImage);
    /* Run 2D FFT on complex values.
    * Map absolute values of complex to pixels and store to file.
    */
    tb_fft2d_inplace(FORWARD_FFT, tb_fft, cpxImg, N);
    cpx_to_img(cpxImg, imImage, N, 1);
    fft_shift(imImage, imImage2, N);
    printf("Write img01-magnitude.ppm\n");
    writeppm("img01-magnitude.ppm", N, N, imImage2);

    /* Run inverse 2D FFT on complex values */
    tb_fft2d_inplace(INVERSE_FFT, tb_fft, cpxImg, N);
    cpx_to_img(cpxImg, imImage, N, 0);
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
        free(cpxImg[i]);
    free(cpxImg);
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
    tb_cpx **cpxImg;

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
    cpxImg = (tb_cpx **)malloc(sizeof(tb_cpx) * N);
    for (i = 0; i < N; ++i)
    {
        cpxImg[i] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
    }

    /* Set real values from image values.
    * Store the real-value version of the image.
    */
    img_to_cpx(image, cpxImg, N);
    cpx_to_img(cpxImg, image, N, 0);
    tb_fft2d(FORWARD_FFT, tb_fft, cpxImg, N);
    tb_fft2d(INVERSE_FFT, tb_fft, cpxImg, N);
    cpx_to_img(cpxImg, imImage, N, 0);
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
        free(cpxImg[i]);
    free(cpxImg);
    free(imImage);
    return res;
}

void prntTrans(tb_cpx **seq, uint32_t N)
{
    uint32_t x, y, w, h;
    w = N < 11 ? N : 10;
    h = N < 11 ? N : 10;
    printf("\n");
    for (y = 0; y < h; ++y) {        
        for (x = 0; x < w; ++x) {
            printf("(%.0f, %.0f) ", seq[y][x].r, seq[y][x].i);
        }
        printf("\n");
    }
}

unsigned char test_transpose(uint32_t N)
{    
    uint32_t x, y;
    tb_cpx **in, **in2;
    in = (tb_cpx **)malloc(sizeof(tb_cpx) * N * N);
    in2 = (tb_cpx **)malloc(sizeof(tb_cpx) * N * N);
    for (y = 0; y < N; ++y) {
        in[y] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        in2[y] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        for (x = 0; x < N; ++x) {
            in[y][x] = { (float)x, (float)y };
            in2[y][x] = { (float)x, (float)y };
        }
    }    
    transpose(in, N);
    transpose_block(in2, N, 16);
    for (y = 0; y < N; ++y) {
        for (x = 0; x < N; ++x) {
            if (in[y][x].r != (float)y || in[x][y].r != (float)x) {
                printf("Error no block at (%u, %u) is: (%f,%f)\n", x, y, in[x][y].r, in[x][y].i);
                return 0;
            }
            if (in2[y][x].r != (float)y || in2[x][y].r != (float)x) {
                printf("Error no block at (%u, %u) is: (%f,%f)\n", x, y, in2[x][y].r, in2[x][y].i);
                return 0;
            }
        }
    }
    return 1;
}

double test_time_transpose(void(*transpose_function)(tb_cpx**, uint32_t), uint32_t N)
{
    LARGE_INTEGER freq, tStart, tStop;
    tb_cpx **in = (tb_cpx **)malloc(sizeof(tb_cpx) * N * N);
    for (uint32_t y = 0; y < N; ++y) {
        in[y] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        for (uint32_t x = 0; x < N; ++x) {
            in[y][x] = { (float)x, (float)y };
        }
    }
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&tStart);
    for (int i = 0; i < NO_TESTS; ++i){
        transpose_function(in, N);
    }
    QueryPerformanceCounter(&tStop);
    return (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (double)freq.QuadPart;
}

double test_time_transpose_block(void(*transpose_function)(tb_cpx**, uint32_t, uint32_t), uint32_t block_size, uint32_t N)
{
    LARGE_INTEGER freq, tStart, tStop;
    uint32_t x, y;
    tb_cpx **in = (tb_cpx **)malloc(sizeof(tb_cpx) * N * N);
    for (y = 0; y < N; ++y) {
        in[y] = (tb_cpx *)malloc(sizeof(tb_cpx) * N);
        for (x = 0; x < N; ++x) {
            in[y][x] = { (float)x, (float)y };
        }
    }
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&tStart);
    for (int i = 0; i < NO_TESTS; ++i){
        transpose_function(in, N, block_size);
    }
    QueryPerformanceCounter(&tStop);
    return (double)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (double)freq.QuadPart;
}

void kiss_fft(double dir, tb_cpx *in, tb_cpx *out, uint32_t N)
{
    kiss_fft_cfg cfg = kiss_fft_alloc(N, (dir == INVERSE_FFT), NULL, NULL);
    kiss_fft(cfg, in, out);
    free(cfg);
}