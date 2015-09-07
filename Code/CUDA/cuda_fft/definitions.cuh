#ifndef TB_DEFINITIONS_H
#define TB_DEFINITIONS_H

#include "cuComplex.h"

typedef const float fftDirection;

typedef void(*fftFunction)(fftDirection direction, cuFloatComplex **in, cuFloatComplex **out, const int n);
typedef void(*transposeFunction)(cuFloatComplex **seq, const int, const int n);
typedef void(*twiddleFunc)(cuFloatComplex *W, const int, const int n);
typedef void(*bitReverseFunction)(cuFloatComplex *seq, const double dir, const int lead, const int n);

#define M_2_PI 6.28318530718f
#define M_PI 3.14159265359f

#define FFT_FORWARD -1.0f
#define FFT_INVERSE 1.0f

#endif