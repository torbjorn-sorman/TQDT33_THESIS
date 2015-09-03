#ifdef _OPENMP
#include <omp.h> 
#endif

#include "tb_transpose.h"

void transpose(cpx **seq, const int b, const int n_threads, const int n)
{
    int x, y;
    cpx tmp;
#pragma omp parallel for schedule(static, n / n_threads) private(y, x, tmp) shared(n, seq)
    for (y = 0; y < n; ++y) {
        for (x = y + 1; x < n; ++x) {
            tmp = seq[y][x];
            seq[y][x] = seq[x][y];
            seq[x][y] = tmp;
        }
    }
}

void transpose_block(cpx **seq, const int b, const int n_threads, const int n)
{
    int blx, bly, x, y;
    cpx tmp;
#pragma omp parallel for schedule(static, (n / b) / n_threads) private(bly, blx, y, x, tmp) shared(b, n, seq)
    for (bly = 0; bly < n; bly += b) {
        for (blx = bly; blx < n; blx += b) {
            for (y = bly; y < b + bly; ++y) {
                for (x = blx; x < b + blx; ++x) {
                    if (x > y) {
                        tmp = seq[y][x];
                        seq[y][x] = seq[x][y];
                        seq[x][y] = tmp;
                    }
                }
            }
        }
    }
}