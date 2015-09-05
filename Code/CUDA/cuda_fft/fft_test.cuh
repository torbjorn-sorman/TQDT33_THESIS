#ifndef FFT_TEST_CUH
#define FFT_TEST_CUH

#include "definitions.cuh"

int checkError(cpx *seq, cpx *ref, const int n, int print);
cpx *get_seq(const int n);
cpx *get_seq(const int n, const int sinus);
cpx *get_seq(const int n, cpx *src);
double avg(double m[], int n);

#endif