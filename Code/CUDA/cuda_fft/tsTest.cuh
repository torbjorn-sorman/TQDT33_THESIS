#ifndef TSTEST_CUH
#define TSTEST_CUH

#include "tsDefinitions.cuh"
#include <Windows.h>

/* Performance measure on Windows, result in micro seconds */

#define NUM_PERFORMANCE 20

void startTimer();
double stopTimer();

void console_print(cpx *seq, const int n);

unsigned int power(const unsigned int base, const unsigned int exp);
unsigned int power2(const int unsigned exp);

int checkError(cpx *seq, cpx *ref, float refScale, const int n, int print);
int checkError(cpx *seq, cpx *ref, const int n, int print);
int checkError(cpx *seq, cpx *ref, const int n);

cpx *get_seq(const int n);
cpx *get_seq(const int n, const int sinus);
cpx *get_seq(const int n, cpx *src);

double avg(double m[], int n);

void fftMalloc(int n, cpx **dev_in, cpx **dev_out, cpx **dev_W, cpx **in, cpx **ref, cpx **out);
int fftResultAndFree(int n, cpx **dev_in, cpx **dev_out, cpx **dev_W, cpx **in, cpx **ref, cpx **out);

#endif