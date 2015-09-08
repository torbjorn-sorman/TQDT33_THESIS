#ifndef TB_DEFINITIONS_H
#define TB_DEFINITIONS_H

#include "cuComplex.h"

typedef cuFloatComplex cpx;
typedef const float fftDirection;

typedef void(*fftFunction)(fftDirection direction, cpx **in, cpx **out, const int n);
typedef void(*transposeFunction)(cpx **seq, const int, const int n);
typedef void(*twiddleFunction)(cpx *W, const int, const int n);
typedef void(*bitReverseFunction)(cpx *seq, const double dir, const int lead, const int n);

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FFT_FORWARD -1.0f
#define FFT_INVERSE 1.0f

#endif