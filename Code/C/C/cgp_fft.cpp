/* Copyright (c) 2010 Steve B. Scott
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cgp_fft.h"

#define TWOPI (2.0 * 3.1415926535897932384626433832795)

//Gold-Rader bit reversal
__inline void bit_reverse(double *re, double *im, int n) {
    int n2, i, loop;
    double tn;

    i = 0;
    for (loop = 1; loop < n - 1; loop++) {
        n2 = n >> 1;
        while (i >= n2) {
            i = i - n2;
            n2 = n2 >> 1;
        }
        i = i + n2;
        if (loop < i) {
            tn = re[loop];
            re[loop] = re[i];
            re[i] = tn;
            tn = im[loop];
            im[loop] = im[i];
            im[i] = tn;
        }
    }

}

/*
wre - write real
wim - write imaginary
rre - read real
rim - read imaginary
n - number of samples
half_n - number of samples divided by 2
stage - current stage being processed
step - step number
step_size - number of points to calculate
e - two pi over number of samples (for twiddle factor)
*/
typedef struct {
    double *wre, *wim;
    double *rre, *rim;
    int n;
    int half_n;
    int stage;
    int step;
    int step_size;
    double e;
} FFT_DF;


/*
FFT Stage: Constant-Geometry
Decimation-in-frequency
Out-of-place
Since there is never any access contention between processes
and each stage completes before starting the next
there is no need for mutual exclusion to protect our data.
*/
__inline void fft_df_openmp(void *vdf) {
    FFT_DF *df;
    void *tmp;
    int tp, nh, ni, j, k, start_seg, end_seg, inc;
    double cf, sf, twf;
    df = (FFT_DF*)vdf;

    //determine our starting twiddle and butterfly calculations
    //calculated based on our position in the current step and stage
    k = df->step * df->step_size;
    start_seg = k;
    end_seg = k + df->step_size;
    ni = k << 1;

    tp = 1 << df->stage;
    inc = (int)(start_seg / tp);
    twf = df->e * inc * tp;
    cf = cos(twf);
    sf = sin(twf);
    for (j = start_seg; j < end_seg; j++) {
        if ((j % tp) == 0) {
            twf = df->e * k;
            cf = cos(twf);
            sf = sin(twf);
        }

        k++;

        nh = j + df->half_n;

        df->wre[ni] = df->rre[j] + df->rre[nh];
        df->wim[ni] = df->rim[j] + df->rim[nh];
        ni++;
        df->wre[ni] = cf*(df->rre[j] - df->rre[nh]) + sf*(-df->rim[j] + df->rim[nh]);
        df->wim[ni] = sf*(df->rre[j] - df->rre[nh]) + cf*(df->rim[j] - df->rim[nh]);
        ni++;
    }
    //swap read/write storage spaces to prep for next stage
    tmp = df->rre;
    df->rre = df->wre;
    df->wre = (double *)tmp;
    tmp = df->rim;
    df->rim = df->wim;
    df->wim = (double *)tmp;

}


void cgp_fft_openmp(double **rei, double **imi, int n, int n_stages, const int no_threads, int i) {
    FFT_DF *df;
    void *tmp;
    int loop, t, n_threads;
    double *re, *im;
    double *rex, *imx; //auxillary storage

    n_threads = no_threads;
    re = *rei;
    im = *imi;

    //allocate auxillary storage space
    rex = (double *)malloc(sizeof(double) * n);
    imx = (double *)malloc(sizeof(double) * n);

    //n threads must be a divisor of n and <= n/2
    if (n_threads > (n >> 1)) {
        n_threads = n >> 1;
    }
    df = (FFT_DF *)malloc(sizeof(FFT_DF) * n_threads);

    for (loop = 0; loop < n_threads; loop++) {
        df[loop].rre = re;
        df[loop].rim = im;
        df[loop].wre = rex;
        df[loop].wim = imx;
        df[loop].n = n;
        df[loop].half_n = n >> 1;
        df[loop].step_size = n / (n_threads * 2);
        df[loop].step = loop;
        df[loop].e = i * TWOPI / n;
    }

    for (loop = 0; loop < n_stages; loop++) {

#pragma omp parallel for schedule(static,1)
        for (t = 0; t < n_threads; t++) {
            df[t].stage = loop;
            fft_df_openmp(&df[t]);
        }

#pragma omp barrier
    }

    if (n_stages & 1) {
        //swap re/im with rex/imx
        tmp = *rei;
        *rei = rex;
        rex = (double *)tmp;
        tmp = *imi;
        *imi = imx;
        imx = (double *)tmp;

        re = *rei;
        im = *imi;
    }

    bit_reverse(re, im, n);

    free(rex);
    free(imx);
}
