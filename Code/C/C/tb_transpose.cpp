#include "tb_transpose.h"

void transpose(tb_cpx **seq, const int b, const int n)
{
    int x, y;
    tb_cpx tmp;
    for (y = 0; y < n; ++y) {
        for (x = y + 1; x < n; ++x) {
            tmp = seq[y][x];
            seq[y][x] = seq[x][y];
            seq[x][y] = tmp;
        }
    }
}

void transpose_openmp(tb_cpx **seq, const int b, const int n)
{
    int x, y;
    tb_cpx tmp;
#pragma omp parallel for private(y, x, tmp) shared(n, seq)
    for (y = 0; y < n; ++y) {
        for (x = y + 1; x < n; ++x) {
            tmp = seq[y][x];
            seq[y][x] = seq[x][y];
            seq[x][y] = tmp;
        }
    }
}

void transpose_block(tb_cpx **seq, const int b, const int n)
{
    int blx, bly, x, y;
    tb_cpx tmp;
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

void transpose_block_openmp(tb_cpx **seq, const int b, const int n)
{
    int blx, bly, x, y;
    tb_cpx tmp;
#pragma omp parallel for private(bly, blx, y, x, tmp) shared(b, n, seq)
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