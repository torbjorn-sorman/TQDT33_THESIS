#ifndef TSTEST_CUH
#define TSTEST_CUH

#include "tsDefinitions.cuh"
#include <Windows.h>

/* Performance measure on Windows, result in micro seconds */

#define NUM_PERFORMANCE 20

void startTimer();
double stopTimer();

void console_print(cpx *seq, cInt n);
void console_print_cpx_img(cpx *seq, cInt n);

unsigned int power(cUInt base, cUInt exp);
unsigned int power2(cUInt exp);

int checkError(cpx *seq, cpx *ref, cFloat refScale, cInt n, cInt print);
int checkError(cpx *seq, cpx *ref, cInt n, cInt print);
int checkError(cpx *seq, cpx *ref, cInt n);

cpx *get_seq(cInt n);
cpx *get_seq(cInt n, cInt sinus);
cpx *get_seq(cInt n, cpx *src);
cpx *get_sin_img(cInt n);

double avg(double m[], cInt n);

void fftMalloc(cInt n, cpx **dev_in, cpx **dev_out, cpx **dev_W, cpx **in, cpx **ref, cpx **out);
int fftResultAndFree(cInt n, cpx **dev_in, cpx **dev_out, cpx **dev_W, cpx **in, cpx **ref, cpx **out);

#endif