#include <Windows.h>
#include <limits>
#include "fftw-3.3.4-dll32\fftw3.h"

#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_test.h"
#include "tb_test_helper.h"
#include "tb_fft.h"
#include "tb_fft_helper.h"
#include "tb_image.h"
#include "tb_transpose.h"
#include "tb_math.h"
#include "tb_print.h"
#include "tb_filter.h"
#include "cgp_fft.h"

#define NO_TESTS 8

LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds, Frequency;

#define QPF QueryPerformanceFrequency
#define QPC QueryPerformanceCounter

#define START_TIME QPF(&Frequency); QPC(&StartingTime)
#define STOP_TIME(RES) QPC(&EndingTime); ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart; ElapsedMicroseconds.QuadPart *= 1000000; ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;(RES) = (double)ElapsedMicroseconds.QuadPart

void validate(fft_func fn, const int n_threads, const unsigned int max_elements)
{
    cpx *in, *out, *ref;
    for (unsigned int n = 4; n <= max_elements; n *= 2) {
        in = get_seq(n, 1);
        out = get_seq(n);
        ref = get_seq(n, in);
        fn(FORWARD_FFT, &in, &out, n_threads, n);
        fn(INVERSE_FFT, &out, &in, n_threads, n);
        checkError(in, ref, n, 1);
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
    printf("\n%s\n", name);
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

/*

unsigned char test_equal_dft(fft_body_fn fn, fft_body_fn ref, const int n_threads, const int inplace)
{
int n;
unsigned char res = 1;
cpx *fft_out, *dft_out, *in, *in2, *W;

for (n = 2; n < MAX_LENGTH; n *= 2) {
W = (cpx *)malloc(sizeof(cpx) * n);
twiddle_factors(W, 32 - log2_32(n), n_threads, n);
in = get_seq(n, 1);
if (inplace == 0)
{
fft_out = get_seq(n);
dft_out = get_seq(n);
fft_template(ref, FORWARD_FFT, in, dft_out, W, n_threads, n);
fft_template(fn, FORWARD_FFT, in, fft_out, W, n_threads, n);
res = checkError(dft_out, fft_out, n, 1);
free(dft_out);
free(fft_out);
}
else
{
in2 = get_seq(n, in);
fft_template(ref, FORWARD_FFT, in, in, W, n_threads, n);
fft_template(fn, FORWARD_FFT, in2, in2, W, n_threads, n);
res = checkError(in, in2, n, 1);
free(W);
free(in2);
}
free(W);
free(in);
}

return res;
}

unsigned char test_equal_dft2d(fft2d_fn fft_2d, fft_body_fn dif, fft_body_fn ref, const int n_threads, const int inplace)
{
int n, m, size;
unsigned char *image, *imImage, *imImageRef;
char filename[30];
cpx **cpxImg, **cpxImgRef;
size = 4096;
sprintf_s(filename, 30, "img/lena/%u.ppm", size);
image = readppm(filename, &n, &m);
imImage = get_empty_img(size, size);
imImageRef = get_empty_img(size, size);
cpxImg = get_seq2d(n);
cpxImgRef = get_seq2d(n);
img_to_cpx(image, cpxImg, size);
img_to_cpx(image, cpxImgRef, size);

fft_2d(dif, FORWARD_FFT, cpxImg, n_threads, size);
fft_2d(ref, FORWARD_FFT, cpxImgRef, n_threads, size);

printf("Max Fw error: %f\n", cpx_diff(cpxImg, cpxImgRef, size));
printf("Avg Fw error: %f\n", cpx_avg_diff(cpxImg, cpxImgRef, size));

fft_2d(dif, INVERSE_FFT, cpxImg, n_threads, size);
fft_2d(ref, INVERSE_FFT, cpxImgRef, n_threads, size);

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

double test_time_dft(fft_body_fn fn, const int n_threads, const int n)
{
int i;
double measures[NO_TESTS];
cpx *in, *out, *W;
in = get_seq(n, 1);
out = get_seq(n);
for (i = 0; i < NO_TESTS; ++i) {
START_TIME;
W = (cpx *)malloc(sizeof(cpx) * n);
twiddle_factors(W, 32 - log2_32(n), n_threads, n);
fft_template(fn, FORWARD_FFT, in, in, W, n_threads, n);
free(W);
STOP_TIME(measures[i]);
}
free(in);
free(out);
return avg(measures, NO_TESTS);
}

double test_time_const_geom(int n_threads, const int n)
{
int i;
double measures[NO_TESTS];
cpx *in, *out, *W;
in = get_seq(n, 1);
out = get_seq(n);
for (i = 0; i < NO_TESTS; ++i) {
START_TIME;
W = (cpx *)malloc(sizeof(cpx) * n);
twiddle_factors(W, n_threads, n);
fft_const_geom(FORWARD_FFT, &in, &out, W, n_threads, n);
free(W);
STOP_TIME(measures[i]);
}
free(in);
free(out);
return avg(measures, NO_TESTS);
}

double test_time_const_geom_no_twiddle(int n_threads, const int n)
{
int i;
double measures[NO_TESTS];
cpx *in, *out;
in = get_seq(n, 1);
out = get_seq(n);
for (i = 0; i < NO_TESTS; ++i) {
START_TIME;
fft_const_geom(FORWARD_FFT, &in, &out, n_threads, n);
STOP_TIME(measures[i]);
}
free(in);
free(out);
return avg(measures, NO_TESTS);
}

double test_time_ext(void(*fn)(const double, cpx *, cpx *, int, const int), const int n_threads, const int n)
{
int i;
double measures[NO_TESTS];
cpx *in, *out;
in = get_seq(n, 1);
out = get_seq(n);
for (i = 0; i < NO_TESTS; ++i) {
START_TIME;
fn(FORWARD_FFT, in, out, n_threads, n);
STOP_TIME(measures[i]);
}
free(in);
free(out);
return avg(measures, NO_TESTS);
}

double test_time_dft_2d(fft2d_fn fft2d, fft_body_fn dif, const int n_threads, const int n)
{
int i, x, y;
char filename[30];
unsigned char *image;
cpx **cpxImg, **cpxImgRef;
double measures[NO_TESTS];

cpxImg = get_seq2d(n);
cpxImgRef = get_seq2d(n);
sprintf_s(filename, 30, "img/splash/%u.ppm", n);
image = readppm(filename, &x, &y);
img_to_cpx(image, cpxImgRef, n);

for (i = 0; i < NO_TESTS; ++i)
{
copy_seq2d(cpxImgRef, cpxImg, n);
START_TIME;
fft2d(dif, FORWARD_FFT, cpxImg, n_threads, n);
fft2d(dif, INVERSE_FFT, cpxImg, n_threads, n);
STOP_TIME(measures[i]);
}
free(image);
free_seq2d(cpxImg, n);
return avg(measures, NO_TESTS);
}

double test_cmp_time(fft_body_fn fn, fft_body_fn ref, const int n_threads)
{
int n;
double time, time_ref, diff, rel, sum, sum_ref;
rel = DBL_MIN;
sum = sum_ref = 0.0;
printf("\trel.\tdiff.\ttime\tref\tN\n");
time = test_time_dft(fn, n_threads, 512);
time_ref = test_time_dft(ref, n_threads, 512);
for (n = 8; n < MAX_LENGTH; n *= 2) {
time = test_time_dft(fn, n_threads, n);
time_ref = test_time_dft(ref, n_threads, n);
diff = time_ref - time;
sum += time;
sum_ref += time_ref;
rel += diff / time_ref;
printf("(ms)\t%.2f\t%.1f\t%.1f\t%.1f\t%u\n", diff / time_ref, diff, time, time_ref, n);

}
return rel / 22;
}

unsigned char test_image(fft2d_fn fft2d, fft_body_fn dif, char *filename, const int n_threads, const int n)
{
int res, w, m, i;
char file[30];
unsigned char *image, *imImage, *imImage2, *greyImage;
cpx **cpxImg;

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
writeppm("img_out/img00-org.ppm", n, n, image);
cpx_to_img(cpxImg, greyImage, n, 0);
printf("Write img00-grey.ppm\n");
writeppm("img_out/img00-grey.ppm", n, n, greyImage);
// Run 2D FFT on complex values.
// Map absolute values of complex to pixels and store to file.

fft2d(dif, FORWARD_FFT, cpxImg, n_threads, n);

// Test to apply filter...
filter_blur(1024, cpxImg, n);

cpx_to_img(cpxImg, imImage, n, 1);
fft_shift(imImage, imImage2, n);
printf("Write img01-magnitude.ppm\n");
writeppm("img_out/img01-magnitude.ppm", n, n, imImage2);

// Run inverse 2D FFT on complex values
fft2d(dif, INVERSE_FFT, cpxImg, n_threads, n);
cpx_to_img(cpxImg, imImage, n, 0);
printf("Write img02-fftToImage.ppm\n");
writeppm("img_out/img02-fftToImage.ppm", n, n, imImage);

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

unsigned char test_transpose(transpose_fn fn, const int b, const int n_threads, const int n)
{
int x, y, res;
cpx **in;
in = get_seq2d(n, 2);
fn(in, b, n_threads, n);
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

double test_time_transpose(transpose_fn trans_fn, const int b, const int n_threads, const int n)
{
int i;
double measures[NO_TESTS];
cpx **in;
in = get_seq2d(n, 2);
for (i = 0; i < NO_TESTS; ++i) {
START_TIME;
trans_fn(in, b, n_threads, n);
STOP_TIME(measures[i]);
}
free_seq2d(in, n);
return avg(measures, NO_TESTS);
}

double test_time_twiddle(twiddle_fn fn, const int n_threads, const int n)
{
int i, lead;
double measures[NO_TESTS];
cpx *w;
lead = 32 - log2_32(n);
w = get_seq(n);
for (i = 0; i < NO_TESTS; ++i) {
START_TIME;
fn(w, lead, n_threads, n);
STOP_TIME(measures[i]);
}
free(w);
return avg(measures, NO_TESTS);
}

unsigned char test_twiddle(twiddle_fn fn, twiddle_fn ref, const int n_threads, const int n)
{
int lead, i, res;
cpx *w, *w_ref;
lead = 32 - log2_32(n);
w = get_seq(n);
w_ref = get_seq(n);

fn(w, lead, n_threads, n);
ref(w_ref, lead, n_threads, n);

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

double test_time_reverse(bit_reverse_fn fn, const int n_threads, const int n)
{
int i, lead;
double measures[NO_TESTS];
cpx *w;
lead = 32 - log2_32(n);
w = get_seq(n);
for (i = 0; i < NO_TESTS; ++i) {
START_TIME;
fn(w, -1, lead, n_threads, n);
STOP_TIME(measures[i]);
}
free(w);
return avg(measures, NO_TESTS);
}

void test_complete_fft(char *name, fft_body_fn fn, const int n_threads)
{
int i;
double tm;
cpx *in, *ref, *W;
printf("\n%s\n", name);

for (i = 4; i < MAX_LENGTH; i *= 2) {
in = get_seq(i);
in[1].r = 1;
ref = get_seq(i, in);
W = (cpx *)malloc(sizeof(cpx) * i);
twiddle_factors(W, 32 - log2_32(i), n_threads, i);
fft_template(fn, FORWARD_FFT, in, in, W, n_threads, i);
twiddle_factors_inverse(W, n_threads, i);
fft_template(fn, INVERSE_FFT, in, in, W, n_threads, i);
free(W);
checkError(in, ref, i, 1);
free(in);
free(ref);
}

FILE *f;
fopen_s(&f, name, "w");

printf("Length\tTime\n");
for (i = 4; i < MAX_LENGTH; i *= 2) {
printf("%d\t%.1f\n", i, tm = test_time_dft(fn, n_threads, i));
fprintf_s(f, "%f\n", tm);
}

fclose(f);
}

void test_complete_fft_cg(char *name, const int n_threads)
{
int i;
double tm;
cpx *in, *out, *ref, *W;

printf("\n%s\n", name);
for (i = 4; i < MAX_LENGTH; i *= 2) {
in = get_seq(i);
in[1].r = 1;
out = get_seq(i);
ref = get_seq(i, in);
W = (cpx *)malloc(sizeof(cpx) * i);
twiddle_factors(W, n_threads, i);
fft_const_geom(FORWARD_FFT, &in, &out, W, n_threads, i);
twiddle_factors_inverse(W, n_threads, i / 2);
fft_const_geom(INVERSE_FFT, &out, &in, W, n_threads, i);
free(W);
checkError(in, ref, i, 1);
free(in);
free(ref);
}

FILE *f;
fopen_s(&f, name, "w");

printf("Length\tTime\n");
for (i = 4; i < MAX_LENGTH; i *= 2) {
printf("%d\t%.1f\n", i, tm = test_time_const_geom(n_threads, i));
fprintf_s(f, "%f\n", tm);
}
fclose(f);
}

void test_complete_fft_cg_no_twiddle(char *name, const int n_threads)
{
int i;
double tm;
cpx *in, *out, *ref;

printf("\n%s\n", name);
for (i = 4; i < MAX_LENGTH; i *= 2) {
in = get_seq(i);
in[1].r = 1;
out = get_seq(i);
ref = get_seq(i, in);

fft_const_geom(FORWARD_FFT, &in, &out, n_threads, i);
fft_const_geom(INVERSE_FFT, &out, &in, n_threads, i);

checkError(in, ref, i, 1);
free(in);
free(ref);
}

FILE *f;
fopen_s(&f, name, "w");

printf("Length\tTime\n");
for (i = 4; i < MAX_LENGTH; i *= 2) {
printf("%d\t%.1f\n", i, tm = test_time_const_geom_no_twiddle(n_threads, i));
fprintf_s(f, "%f\n", tm);
}
fclose(f);
}


void test_complete_ext(char *name, void(*fn)(const double, cpx *, cpx *, int, const int), const int n_threads)
{
int i, n;
double tm;
cpx *in, *out, *ref;
n = 16;
in = get_seq(n);
in[1].r = 1;
out = get_seq(n);
ref = get_seq(n, in);
printf("\n%s\n", name);

fn(FORWARD_FFT, in, out, n_threads, n);
///*
console_newline(1);
console_print(out, n);
console_newline(1);

fn(INVERSE_FFT, in, out, n);

checkError(in, ref, n, 1);

console_separator();
console_print(ref, n);
console_newline();
console_print(in, n);
console_separator();
//
free(in);
free(out);
free(ref);

FILE *f;
fopen_s(&f, name, "w");

printf("Length\tTime\n");
for (i = 4; i < MAX_LENGTH; i *= 2) {
printf("%d\t%.1f\n", i, tm = test_time_ext(fn, n_threads, i));
fprintf_s(f, "%f\n", tm);
}
fclose(f);
}

void test_complete_fft2d(char *name, const int n_threads, fft2d_fn fn)
{
int n, i;
cpx **in, **ref;
fft_body_fn body;
body = fft_body;
n = 512;
in = get_seq2d(n, 1);
ref = get_seq2d(n, in);

fn(body, FORWARD_FFT, in, n_threads, n);
fn(body, INVERSE_FFT, in, n_threads, n);
printf("\n%s\n", name);
checkError(in, ref, n, 1);

printf("Length\tTime\n");
for (i = 4; i < 4096; i *= 2) {
printf("%d\t%.1f\n", i, test_time_dft_2d(fn, body, n_threads, i));
}

free(in);
free(ref);
}
*/

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