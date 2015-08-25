#include "tb_transpose.h"

void transpose(tb_cpx **seq, uint32_t n)
{
    uint32_t x, y;
    tb_cpx tmp;
    for (y = 0; y < n; ++y) {
        for (x = y + 1; x < n; ++x) {
            tmp = seq[y][x];
            seq[y][x] = seq[x][y];
            seq[x][y] = tmp;
        }
    }
}

void transpose_block(tb_cpx **seq, const uint32_t size, const uint32_t block_size)
{
    uint32_t blx, bly, x, y;
    tb_cpx tmp;
    for (bly = 0; bly < size; bly += block_size) {
        for (blx = bly; blx < size; blx += block_size) {
            for (y = bly; y < block_size + bly; ++y) {
                for (x = blx; x < block_size + blx; ++x) {
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