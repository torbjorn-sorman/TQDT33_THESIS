#include <stdio.h>
#include <Windows.h>
#include <cmath>

#include "kiss_fft.h"

#define M_2_PI 6.28318530718
#define N 8192

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef kiss_fft_cpx complex;
/*
struct complex
{
	float r;
	float i;
};
*/

static const int tab32[32] = 
{ 
    0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31 
};

static const uint32_t revTbl256[] =
{
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
    0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
    0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
    0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
    0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
    0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
    0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
    0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
    0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
    0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
    0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
    0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
    0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
    0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
    0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
    0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};

int log2_32(uint32_t value)
{
	value |= value >> 1; value |= value >> 2; value |= value >> 4; value |= value >> 8; value |= value >> 16;
	return tab32[(uint32_t)(value * 0x07C4ACDD) >> 27];
}

uint32_t reverse(uint32_t x, uint32_t l)
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    return((x >> 16) | (x << 16)) >> (32 - l);
}

#define rev(X,L) ((revTbl256[X & 0xff] << 24) | (revTbl256[(X >> 8) & 0xff] << 16) | (revTbl256[(X >> 16) & 0xff] << 8) | (revTbl256[(X >> 24) & 0xff])) >> L
#define C_MUL_RE(A,B) A.r * B.r + A.i * B.i
#define C_MUL_IM(A,B) A.r * B.i + A.i * B.r

// Slightly faster... but only by small margin.
uint32_t revTbl32(uint32_t v, uint32_t l)
{
    return ((revTbl256[v & 0xff] << 24) | (revTbl256[(v >> 8) & 0xff] << 16) | (revTbl256[(v >> 16) & 0xff] << 8) | (revTbl256[(v >> 24) & 0xff])) >> (32 - l);
}

void naive_dft(complex *x, complex *X);
void fft(complex *x, complex *X);
void printTime(LARGE_INTEGER tStart, LARGE_INTEGER tStop, LARGE_INTEGER freq);
void printResult(complex *c, int n, char *str, int verified);
int verify_impulse(complex *c, int size);
void compareComplex(complex *c1, complex *c2, float t1, float t2);
void test_p();

int main()
{
	LARGE_INTEGER freq, tStart, tStop;
    double time_FFT, time_KFFT, time_NDFT;

	/* Get ticks per second */
	QueryPerformanceFrequency(&freq);
    
	/* Prep data */
    complex *impulse = (complex *)malloc(sizeof(complex)*N);
    complex *res_FFT = (complex *)malloc(sizeof(complex)*N);
    complex *cx_out = (complex *)malloc(sizeof(complex)*N);
	for (int i = 0; i < N; ++i)
	{
		impulse[i].r = 0.f;
        impulse[i].i = 0.f;
	}
	impulse[1].r = 1.0;
        
    /* FFT */
    printf("Running FFT...\n");
    fft(impulse, res_FFT);
    QueryPerformanceCounter(&tStart);
    fft(impulse, res_FFT);
    QueryPerformanceCounter(&tStop);
    printTime(tStart, tStop, freq);
    time_FFT = (float)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart;

    /* KissFFT for comparissons */
    printf("Running KISS_FFT...\n");
    kiss_fft_cfg cfgW = kiss_fft_alloc(N, 0, 0, 0);
    kiss_fft(cfgW, impulse, cx_out);
    QueryPerformanceCounter(&tStart);
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, 0, 0);
    kiss_fft(cfg, impulse, cx_out);
    QueryPerformanceCounter(&tStop);
    printTime(tStart, tStop, freq);
    time_KFFT = (float)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart;
    	
	//printResult(res_FFT, N, "impulse", verify_impulse(res_FFT, N));
    //printResult(cx_out, N, "impulse", verify_impulse(cx_out, N));
	compareComplex(res_FFT, cx_out, time_FFT, time_KFFT);
	
    free(cfg);
	free(impulse);
	free(res_FFT);
	getchar();
	return 0;
}

/* Naive Discrete Fourier Transform, essentially as per definition */
void naive_dft(complex *x, complex *X)
{
	float real, img;
	complex y = { 0.0, 0.0 };
	float re, im;
	complex tmp = { 0.0, 0.0 };
	float theta = 1.0;
	float c1 = -M_2_PI / N;
	float c2 = 1.0;
	for (int k = 0; k < N; ++k)
	{
		real = 0.0;
		img = 0.0;
		c2 = c1 * k;
		for (int n = 0; n < N; ++n)
		{
			theta = c2 * n;
			re = cos(theta);
			im = sin(theta);
			real += x[n].r * re + x[n].i * im;
			img += x[n].r * im + x[n].i * re;
		}
		x[k].r = real;
		x[k].i = img;
	}
}

/* Naive Fast Fourier Transform */
void fft(complex *x, complex *X)
{
    complex *tmp = (complex *)malloc(sizeof(complex)*N);
    complex *W = (complex *)malloc(sizeof(complex)*N);
	const uint32_t depth = log2_32(N);
	const uint32_t n_half = N / 2;
	float w_angle = -M_2_PI / N;
	float theta;
    uint32_t trail = 32 - depth;
	uint32_t bit = 0;
	float u_re, u_im, l_re, l_im;
	uint32_t u, l, p;
	uint32_t dist, dist_2, offset;
	for (uint32_t n = 0; n < N; ++n)
	{
		tmp[n] = x[n];
        /*
        p = n;
        p = (((p & 0xaaaaaaaa) >> 1) | ((p & 0x55555555) << 1));
        p = (((p & 0xcccccccc) >> 2) | ((p & 0x33333333) << 2));
        p = (((p & 0xf0f0f0f0) >> 4) | ((p & 0x0f0f0f0f) << 4));
        p = (((p & 0xff00ff00) >> 8) | ((p & 0x00ff00ff) << 8));
        theta = w_angle * (((p >> 16) | (p << 16)) >> trail);
        */
        theta = w_angle * (rev(n, trail));
		W[n].r = cos(theta);
		W[n].i = sin(theta);
	}

	dist = N;
	for (uint32_t k = 0; k < depth; ++k)
	{
		bit = depth - 1 - k;
		dist_2 = dist;
		dist = dist >> 1;
		offset = 0;
		for (uint32_t n = 0; n < n_half; ++n)
		{
			offset += (n >= (dist + offset)) * dist_2;
			l = (n & ~(1 << bit)) + offset;
			u = l + dist;
			// Lower			
            p = (l >> bit);
            l_re = tmp[l].r;
            l_im = tmp[l].i;
            tmp[l].r = l_re + C_MUL_RE(W[p], tmp[u]);
            tmp[l].i = l_im + C_MUL_IM(W[p], tmp[u]);
			// Upper
            p = (u >> bit);
            u_re = l_re + C_MUL_RE(W[p], tmp[u]);
            tmp[u].i = l_im + C_MUL_IM(W[p], tmp[u]);
            tmp[u].r = u_re;
		}
	}
	for (int n = 0; n < N; ++n)
	{
		X[n] = tmp[rev(n, trail)];
	}
	free(tmp);
	free(W);
}

void printTime(LARGE_INTEGER tStart, LARGE_INTEGER tStop, LARGE_INTEGER freq)
{
	printf("Time (ms): %f\n", (float)(tStop.QuadPart - tStart.QuadPart) * 1000.0 / (float)freq.QuadPart);
}

void printResult(complex *c, int n, char *str, int verified)
{
	int len = n > 16 ? 16 : n;
	printf("\nResult %s:\n", str);
	for (int i = 0; i < len; ++i)
		printf("{ %f,\t%fi }\n", c[i].r, c[i].i);
	if (len != n)
		printf("...\n");
	printf("\n%s\n", verified ? "Successful" : "Error");
}

void compareComplex(complex *c1, complex *c2, float t1, float t2)
{
	int res = 0;
    double m = 0.00001;
	for (int i = 0; i < N; ++i)
	{
        if ((abs(c1[i].r - c2[i].r) > m) || (abs(c1[i].i - c2[i].i) > m))
		{
			res = 1;
			break;
		}
	}
	printf("\n%s\n", res != 1 ? "EQUAL" : "NOT EQUAL");
}