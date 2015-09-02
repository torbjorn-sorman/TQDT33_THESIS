#ifndef CGP_FFT_H
#define CGP_FFT_H

void cgp_fft_openmp(double **rei, double **imi, int n, int n_stages, int n_threads, int i);

#endif