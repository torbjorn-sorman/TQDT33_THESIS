
#include "tb_filter.h"
#include "tb_fft_helper.h"

#include <cmath>

void filter_edge(const int val, cpx **seq, const int n)
{
    int x, y;
    for (y = 0; y < val; ++y) {
        for (x = 0; x < val; ++x) {
            seq[y][x] = { 0.f, 0.f };
        }
        for (x = n - val; x < n; ++x) {
            seq[y][x] = { 0.f, 0.f };
        }
    }
    for (y = n - val; y < n; ++y) {
        for (x = 0; x < val; ++x) {
            seq[y][x] = { 0.f, 0.f };
        }
        for (x = n - val; x < n; ++x) {
            seq[y][x] = { 0.f, 0.f };
        }
    }
}

void filter_blur(const int val, cpx **seq, const int n)
{
    int x, y, e;
    e = n - val;
    for (y = val; y < e; ++y) {
        for (x = val; x < e; ++x) {
            seq[y][x] = { 0.f, 0.f };
        }     
    }
}


cpx action(cpx e)
{
    return{ 0.f, 0.f };
}

void do_filter(cpx **seq, int n)
{
    int x, y, r, xr, mr;
    mr = n / 4;
    r = n > mr ? mr : n;
    for (y = 0; y < r; ++y) {
        xr = (int)round(sqrt(r * r - y * y));
        for (x = 0; x < xr; ++x) {
            seq[y][x] = action(seq[y][x]);
        }
        for (x = n - xr; x < n; ++x) {
            seq[y][x] = action(seq[y][x]);
        }
    }
    for (y = n - r; y < n; ++y) {
        xr = (int)round(sqrt(r * r - (n - y) * (n - y)));
        for (x = 0; x < xr; ++x) {
            seq[y][x] = action(seq[y][x]);
        }
        for (x = n - xr; x < n; ++x) {
            seq[y][x] = action(seq[y][x]);
        }
    }
}

int within(float r, cpx p, cpx ref)
{
    float x, y;
    x = p.r - ref.r;
    y = p.i - ref.i;
    return sqrt(x*x + y*y) <= r;
}

void do_another_filter(cpx **seq, int n_threads, int n)
{
    int x, y;
    float radius;
    cpx center;
    center = { n * 0.5f, n * 0.5f };
    radius = n * 0.15f;

    fft_shift(seq, n_threads, n);
    for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
            if (within(radius, seq[y][x], center)) {
                seq[y][x] = { 0.0, 0.0 };
            }
        }
    }
    fft_shift(seq, n_threads, n);
}