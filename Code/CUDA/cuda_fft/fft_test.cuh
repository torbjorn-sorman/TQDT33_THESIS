#ifndef FFT_TEST_CUH
#define FFT_TEST_CUH

#include "definitions.cuh"

int checkError(cuFloatComplex *seq, cuFloatComplex *ref, float refScale, const int n, int print);
int checkError(cuFloatComplex *seq, cuFloatComplex *ref, const int n, int print);
int checkError(cuFloatComplex *seq, cuFloatComplex *ref, const int n);

cuFloatComplex *get_seq(const int n);
cuFloatComplex *get_seq(const int n, const int sinus);
cuFloatComplex *get_seq(const int n, cuFloatComplex *src);
double avg(double m[], int n);

#endif