#ifndef TSTEST_CUH
#define TSTEST_CUH

#include "tsDefinitions.cuh"
#include <Windows.h>
#include "tsHelper.cuh"

/* Performance measure on Windows, result in micro seconds */

void startTimer();
double stopTimer();

void console_print(cpx *seq, int n);
void console_print_cpx_img(cpx *seq, int n);

unsigned int power(unsigned int base, unsigned int exp);
unsigned int power2(unsigned int exp);

int checkError(cpx *seq, cpx *ref, float refScale, int n, int print);
int checkError(cpx *seq, cpx *ref, int n, int print);
int checkError(cpx *seq, cpx *ref, int n);

cpx *get_seq(int n);
cpx *get_seq(int n, int sinus);
cpx *get_seq(int n, cpx *src);
cpx *get_sin_img(int n);

double avg(double m[], int n);

void fftMalloc(int n, cpx **dev_in, cpx **dev_out, cpx **dev_W, cpx **in, cpx **ref, cpx **out);
int fftResultAndFree(int n, cpx **dev_in, cpx **dev_out, cpx **dev_W, cpx **in, cpx **ref, cpx **out);
void fft2DSetup(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o, size_t *size, char *image_name, int sinus, int n);
void fft2DShakedown(cpx **in, cpx **ref, cpx **dev_i, cpx **dev_o);
int fft2DCompare(cpx *in, cpx *ref, cpx *dev, size_t size, int len);

#endif