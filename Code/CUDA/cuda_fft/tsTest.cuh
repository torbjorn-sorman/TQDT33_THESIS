#ifndef TSTEST_CUH
#define TSTEST_CUH

#include "tsDefinitions.cuh"
#include <Windows.h>

/* Performance measure on Windows, result in micro seconds */

#define NUM_PERFORMANCE 20

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

#endif