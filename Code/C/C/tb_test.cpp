#include "tb_test.h"

#define NO_TESTS 16
#define NO_RUNS 1

LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds, Frequency;

#define QPF QueryPerformanceFrequency
#define QPC QueryPerformanceCounter

#define START_TIME QPF(&Frequency); QPC(&StartingTime)
#define STOP_TIME(RES) QPC(&EndingTime); ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart; ElapsedMicroseconds.QuadPart *= 1000000; ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;(RES) = (double)ElapsedMicroseconds.QuadPart

void validate(fft_func fn, const int n_threads, const unsigned int max_elements)
{
    cpx *in, *out, *ref;
    for (unsigned int n = 4; n <= max_elements; n *= 2) {
        // Test sinus
        in = get_seq(n, 1);
        out = get_seq(n);
        ref = get_seq(n, in);
        fn(FORWARD_FFT, &in, &out, n_threads, n);
        fn(INVERSE_FFT, &out, &in, n_threads, n);
        if (checkError(in, ref, n, 0)) { printf("Sinus:\t\t"); checkError(in, ref, n, 1); }
        free(in);
        free(out);
        free(ref);
        // Test impulse
        in = get_seq(n);
        in[1].r = 1.f;
        out = get_seq(n);
        ref = get_seq(n, in);
        fn(FORWARD_FFT, &in, &out, n_threads, n);
        fn(INVERSE_FFT, &out, &in, n_threads, n);
        if (checkError(in, ref, n, 0)) { printf("Impulse:\t"); checkError(in, ref, n, 1); }
        free(in);
        free(out);
        free(ref);
    }
}

void validate(fft2d_func fn, const int n_threads, const unsigned int max_elements)
{
    cpx **in, **ref;
    for (unsigned int n = 4; n <= max_elements; n *= 2) {
        in = get_seq2d(n, 1);
        ref = get_seq2d(n, in);
        fn(FORWARD_FFT, in, n_threads, n);
        fn(INVERSE_FFT, in, n_threads, n);
        checkError(in, ref, n, 1);
        free(in);
        free(ref);
    }
}

double timing(fft_func fn, const int n_threads, const int n)
{
    double measures[NO_TESTS];
    cpx *in, *out;
    in = get_seq(n, 1);
    out = get_seq(n);
    for (int i = 0; i < NO_TESTS; ++i) {
        START_TIME;
        for (int j = 0; j < NO_RUNS; ++j)
            fn(FORWARD_FFT, &in, &out, n_threads, n);
        STOP_TIME(measures[i]);
    }
    free(in);
    free(out);
    return avg(measures, NO_TESTS);
}

double timing(fft2d_func fn, const int n_threads, const int n)
{
    double measures[NO_TESTS];
    cpx **in;
    in = get_seq2d(n, 1);
    for (int i = 0; i < NO_TESTS; ++i) {
        START_TIME;
        fn(FORWARD_FFT, in, n_threads, n);
        STOP_TIME(measures[i]);
    }
    free_seq2d(in, n);
    return avg(measures, NO_TESTS);
}

double timing(twiddle_func fn, const int n_threads, const int n)
{
    double measures[NO_TESTS];
    cpx *W;
    W = get_seq(n, 1);
    for (int i = 0; i < NO_TESTS; ++i) {
        START_TIME;
        fn(W, FORWARD_FFT, n);
        STOP_TIME(measures[i]);
    }
    free(W);
    return avg(measures, NO_TESTS);
}

void mtime(char *name, fft_func fn, const int n_threads, int file, const unsigned int max_elements)
{
    double time;
    if (file) {
        char filename[64] = "";
        FILE *f;
        strcat_s(filename, "out/");
        strcat_s(filename, name);
        strcat_s(filename, ".txt");
        fopen_s(&f, filename, "w");
        printf("Length\tTime\n");
        for (unsigned int n = 4; n <= max_elements; n *= 2) {
            printf("%d\t%.1f\n", n, time = timing(fn, n_threads, n));
            fprintf_s(f, "%0.f\n", time);
        }
        printf("Filename: %s\n", filename);
        fclose(f);
    }
    else {
        for (unsigned int n = 4; n <= max_elements; n *= 2) {
            timing(fn, n_threads, n);
        }
    }
}

void mtime(char *name, fft2d_func fn, const int n_threads, int file, const unsigned int max_elements)
{
    double time;
    if (file) {
        char filename[64] = "";
        FILE *f;
        strcat_s(filename, "out/");
        strcat_s(filename, name);
        strcat_s(filename, ".txt");
        fopen_s(&f, filename, "w");
        printf("Length\tTime\n");
        for (unsigned int n = 4; n <= max_elements; n *= 2) {
            printf("%d\t%.1f\n", n, time = timing(fn, n_threads, n));
            fprintf_s(f, "%0.f\n", time);
        }
        printf("Filename: %s\n", filename);
        fclose(f);
    }
    else {
        for (unsigned int n = 4; n <= max_elements; n *= 2) {
            timing(fn, n_threads, n);
        }
    }
}

void mtime(char *name, twiddle_func fn, const int n_threads, int file, const unsigned int max_elements)
{
    double time;
    if (file) {
        char filename[64] = "";
        FILE *f;
        strcat_s(filename, "out/");
        strcat_s(filename, name);
        strcat_s(filename, ".txt");
        fopen_s(&f, name, "w");
        printf("Length\tTime\n");
        for (unsigned int n = 4; n <= max_elements; n *= 2) {
            printf("%d\t%.1f\n", n, time = timing(fn, n_threads, n));
            fprintf_s(f, "%0.f\n", time);
        }
        printf("Filename: %s\n", filename);
        fclose(f);
    }
    else {
        for (unsigned int n = 4; n <= max_elements; n *= 2) {
            timing(fn, n_threads, n);
        }
    }
}

void test_fft(char *name, fft_func fn, const int n_threads, int file, unsigned int max_elem)
{
    if (file) printf("\n%s\n", name);
    validate(fn, n_threads, max_elem);
    mtime(name, fn, n_threads, file, max_elem);
}

void test_short_fft(fft_func fn, const int n_threads, unsigned int max_elem)
{
    double measure;
    cpx *in, *out;
    for (unsigned int n = 4; n <= max_elem; n *= 2) {
        in = get_seq(n, 1);
        out = get_seq(n);
        START_TIME;
        for (int i = 0; i < 100000; ++i) {
            fn(FORWARD_FFT, &in, &out, n_threads, n);
        }
        STOP_TIME(measure);
        free(in);
        free(out);
        printf("Time short %d\t%f\n", n, measure);
    }
}

void test_short_fftw(unsigned int max_elem)
{
    double measure;
    fftw_complex *fftw_in, *fftw_out;
    fftw_plan p;
    for (unsigned int n = 4; n <= max_elem; n *= 2) {
        fftw_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
        fftw_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
        p = fftw_plan_dft_1d(n, fftw_in, fftw_out, FFTW_FORWARD, FFTW_MEASURE);
        START_TIME;
        for (int i = 0; i < 100000; ++i) {
            fftw_execute(p);
        }
        STOP_TIME(measure);
        fftw_destroy_plan(p);
        fftw_free(fftw_in);
        fftw_free(fftw_out);
        printf("Time short %d\t%f\n", n, measure);
    }
}

void test_fft2d(char *name, fft2d_func fn, const int n_threads, int file, unsigned int max_elem)
{
    printf("\n%s\n", name);
    validate(fn, n_threads, max_elem);
    mtime(name, fn, n_threads, file, max_elem);
}

double timing_fftw(const int n)
{
    double measures[NO_TESTS];
    fftw_complex *fftw_in, *fftw_out;
    fftw_plan p;
    fftw_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
    fftw_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
    p = fftw_plan_dft_1d(n, fftw_in, fftw_out, FFTW_FORWARD, FFTW_MEASURE);

    for (int i = 0; i < NO_TESTS; ++i) {
        START_TIME;
        for (int j = 0; j < NO_RUNS; ++j)
            fftw_execute(p);
        STOP_TIME(measures[i]);
    }

    fftw_destroy_plan(p);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
    return avg(measures, NO_TESTS);
}

double timing_fftw2d(const int n)
{
    double measures[NO_TESTS];
    fftw_complex *fftw_in, *fftw_out;
    fftw_plan p;
    fftw_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n * n);
    fftw_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n * n);
    p = fftw_plan_dft_2d(n, n, fftw_in, fftw_out, FFTW_FORWARD, FFTW_MEASURE);

    for (int i = 0; i < NO_TESTS; ++i) {
        START_TIME;
        fftw_execute(p);
        STOP_TIME(measures[i]);
    }

    fftw_destroy_plan(p);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
    return avg(measures, NO_TESTS);
}

void test_fftw(unsigned int max_elem)
{
    double time;
    FILE *f;
    fopen_s(&f, "out/fftw.txt", "w");
    printf("Length\tTime\n");
    for (unsigned int n = 4; n <= max_elem; n *= 2) {
        printf("%d\t%.1f\n", n, time = timing_fftw(n));
        fprintf_s(f, "%0.f\n", time);
    }
    printf("Filename: %s\n", "out/fftw.txt");
    fclose(f);
}

void test_fftw2d(unsigned int max_elem)
{
    double time;
    FILE *f;
    fopen_s(&f, "out/FFTW2D.txt", "w");
    printf("Length\tTime\n");
    for (unsigned int n = 4; n <= max_elem; n *= 2) {
        printf("%d\t%.1f\n", n, time = timing_fftw2d(n));
        fprintf_s(f, "%0.f\n", time);
    }
    printf("Filename: %s\n", "out/FFTW2D.txt");
    fclose(f);
}

const wchar_t *GetWC(const char *c)
{
    size_t res = 0;
    size_t cSize = strlen(c) + 1;
    wchar_t* wc = new wchar_t[cSize];
    mbstowcs_s(&res, wc, 32, c, cSize);
    return wc;
}

BOOL DirectoryExists(LPCTSTR szPath)
{
    DWORD dwAttrib = GetFileAttributes(szPath);

    return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
        (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

int test_image(fft2d_func fft2d, char *filename, const int n_threads, const int n)
{
    int res, w, m, i;
    char file[30];
    char outfile[32];
    unsigned char *image, *imImage, *imImage2, *greyImage;
    cpx **cpxImg;

    sprintf_s(outfile, 50, "out/img/%s", filename);
    if (!DirectoryExists(GetWC(outfile))) {
        printf("Attempt to create %s\n", outfile);
        CreateDirectory(GetWC(outfile), NULL);
    }
    else {
        printf("Directory %s exists\n", outfile);
    }

    sprintf_s(file, 30, "img/%s/%u.ppm", filename, n);
    printf("Read: %s\n", file);
    image = readppm(file, &w, &m);

    if (w != n || m != n)
        return 0;

    greyImage = get_empty_img(n, n);
    imImage = get_empty_img(n, n);
    imImage2 = get_empty_img(n, n);
    cpxImg = get_seq2d(n);

    // Set real values from image values.
    // Store the real-value version of the image.


    img_to_cpx(image, cpxImg, n);
    printf("1 - original.ppm\n");
    sprintf_s(outfile, 50, "out/img/%s/0 - original.ppm", filename);
    writeppm(outfile, n, n, image);
    cpx_to_img(cpxImg, greyImage, n, 0);
    printf("1 - grey.ppm\n");
    sprintf_s(outfile, 50, "out/img/%s/1 - grey.ppm", filename);
    writeppm(outfile, n, n, greyImage);
    // Run 2D FFT on complex values.
    // Map absolute values of complex to pixels and store to file.

    fft2d(FORWARD_FFT, cpxImg, n_threads, n);

    // Test to apply filter...
    filter_blur(1024, cpxImg, n);

    cpx_to_img(cpxImg, imImage, n, 1);
    fft_shift(imImage, imImage2, n);
    printf("2 - magnitude.ppm\n");
    sprintf_s(outfile, 50, "out/img/%s/2 - magnitude.ppm", filename);
    writeppm(outfile, n, n, imImage2);

    // Run inverse 2D FFT on complex values
    fft2d(INVERSE_FFT, cpxImg, n_threads, n);
    cpx_to_img(cpxImg, imImage, n, 0);
    printf("3 - result.ppm\n");
    sprintf_s(outfile, 50, "out/img/%s/3 - result.ppm", filename);
    writeppm(outfile, n, n, imImage);

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

void kiss_fft(double dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    kiss_fft_cfg cfg = kiss_fft_alloc(n, (dir == INVERSE_FFT), NULL, NULL);
    kiss_fft(cfg, *in, *out);
    free(cfg);
}

void cgp_fft(double dir, cpx **in, cpx **out, const int n_threads, const int n)
{
    double *re, *im;
    int i;
    re = (double *)malloc(sizeof(double) * n);
    im = (double *)malloc(sizeof(double) * n);
    for (i = 0; i < n; ++i) {
        re[i] = (double)(*in)[i].r;
        im[i] = (double)(*in)[i].i;
    }

    cgp_fft_openmp(&re, &im, n, log2_32(n), n_threads, (int)dir);

    for (i = 0; i < n; ++i) {
        (*out)[i].r = (float)re[i];
        (*out)[i].i = (float)im[i];
    }

    free(re);
    free(im);
}