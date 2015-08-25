#include "wiki_fft.h"

#include "math.h"

const double TwoPi = 6.283185307179586;

void wiki_fft(double *AVal, double *FTvl, int Nvl, int Nft)
{
    int i, j, n, m, Mmax, Istp;
    double Tmpr, Tmpi, Wtmp, Theta;
    double Wpr, Wpi, Wr, Wi;
    double *Tmvl;

    n = Nvl * 2;
    Tmvl = new double[n + 1];

    for (i = 0; i < Nvl; i++) {
        j = i * 2;
        Tmvl[j] = 0;
        Tmvl[j + 1] = AVal[i];
    }

    i = 1; j = 1;
    while (i < n) {
        if (j > i) {
            Tmpr = Tmvl[i];
            Tmvl[i] = Tmvl[j];
            Tmvl[j] = Tmpr;
            Tmpr = Tmvl[i + 1];
            Tmvl[i + 1] = Tmvl[j + 1];
            Tmvl[j + 1] = Tmpr;
        }
        i = i + 2; 
        m = Nvl;
        while ((m >= 2) && (j > m)) {
            j = j - m;
            m = m >> 2;
        }
        j = j + m;
    }
    Mmax = 2;
    while (n > Mmax)
    {
        Theta = -TwoPi / Mmax;
        Wpi = sin(Theta);
        Wtmp = sin(Theta / 2);
        Wpr = Wtmp * Wtmp * 2;
        Istp = Mmax * 2;
        Wr = 1;
        Wi = 0;
        m = 1;
        while (m < Mmax)
        {
            i = m;
            m = m + 2;
            Tmpr = Wr;
            Tmpi = Wi;
            Wr = Wr - Tmpr * Wpr - Tmpi * Wpi;
            Wi = Wi + Tmpr * Wpi - Tmpi * Wpr;
            while (i < n)
            {
                j = i + Mmax;
                Tmpr = Wr * Tmvl[j] - Wi * Tmvl[j + 1];
                Tmpi = Wi * Tmvl[j] + Wr * Tmvl[j + 1];
                Tmvl[j] = Tmvl[i] - Tmpr;
                Tmvl[j + 1] = Tmvl[i + 1] - Tmpi;
                Tmvl[i] = Tmvl[i] + Tmpr;
                Tmvl[i + 1] = Tmvl[i + 1] + Tmpi;
                i = i + Istp;
            }
        }

        Mmax = Istp;
    }

    for (i = 1; i < Nft; i++) {
        j = i * 2;
        FTvl[i] = sqrt(Tmvl[j] * Tmvl[j] + Tmvl[j + 1] * Tmvl[j + 1]);
    }

    delete[]Tmvl;
}