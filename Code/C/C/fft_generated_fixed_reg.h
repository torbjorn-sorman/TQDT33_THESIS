#ifndef FFT_GENERATED_FIXED_REG_H
#define FFT_GENERATED_FIXED_REG_H

#include "tb_definitions.h"
#include "tb_fft_helper.h"

#define GENERATED_FIXED_REG

__inline static void fft_x4(cpx *in, cpx *out)
{
	cpx in0 = in[0];
	cpx in1 = in[1];
	cpx in2 = in[2];
	cpx in3 = in[3];
	out[0] = cpxMul(cpxAdd(cpxAdd(in0, in2), cpxAdd(in0, in2)), make_cpx(1.000000, 0.0));
	out[1] = cpxMul(cpxAdd(cpxMul(cpxSub(in1, in3), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in1, in3), make_cpx(-0.000000, -1.000000))), make_cpx(1.000000, 0.0));
	out[2] = cpxMul(cpxMul(cpxSub(cpxAdd(in0, in2), cpxAdd(in0, in2)), make_cpx(1.000000, -0.000000)), make_cpx(1.000000, 0.0));
	out[3] = cpxMul(cpxMul(cpxSub(cpxMul(cpxSub(in1, in3), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in1, in3), make_cpx(-0.000000, -1.000000))), make_cpx(1.000000, -0.000000)), make_cpx(1.000000, 0.0));
}

__inline static void fft_x4inv(cpx *in, cpx *out)
{
	cpx in0 = in[0];
	cpx in1 = in[1];
	cpx in2 = in[2];
	cpx in3 = in[3];
	out[0] = cpxMul(cpxAdd(cpxAdd(in0, in2), cpxAdd(in0, in2)), make_cpx(0.250000, 0.0));
	out[1] = cpxMul(cpxAdd(cpxMul(cpxSub(in1, in3), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in1, in3), make_cpx(-0.000000, 1.000000))), make_cpx(0.250000, 0.0));
	out[2] = cpxMul(cpxMul(cpxSub(cpxAdd(in0, in2), cpxAdd(in0, in2)), make_cpx(1.000000, 0.000000)), make_cpx(0.250000, 0.0));
	out[3] = cpxMul(cpxMul(cpxSub(cpxMul(cpxSub(in1, in3), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in1, in3), make_cpx(-0.000000, 1.000000))), make_cpx(1.000000, 0.000000)), make_cpx(0.250000, 0.0));
}

__inline static void fft_x8(cpx *in, cpx *out)
{
	cpx in0 = in[0];
	cpx in1 = in[1];
	cpx in2 = in[2];
	cpx in3 = in[3];
	cpx in4 = in[4];
	cpx in5 = in[5];
	cpx in6 = in[6];
	cpx in7 = in[7];
	out[0] = cpxMul(cpxAdd(cpxAdd(cpxAdd(in0, in4), cpxAdd(in1, in5)), cpxAdd(cpxAdd(in0, in4), cpxAdd(in1, in5))), make_cpx(1.000000, 0.0));
	out[1] = cpxMul(cpxAdd(cpxAdd(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, -0.707107))), cpxAdd(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, -0.707107)))), make_cpx(1.000000, 0.0));
	out[2] = cpxMul(cpxAdd(cpxMul(cpxSub(cpxAdd(in0, in4), cpxAdd(in1, in5)), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(cpxAdd(in0, in4), cpxAdd(in1, in5)), make_cpx(-0.000000, -1.000000))), make_cpx(1.000000, 0.0));
	out[3] = cpxMul(cpxAdd(cpxMul(cpxSub(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, -0.707107))), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, -0.707107))), make_cpx(-0.000000, -1.000000))), make_cpx(1.000000, 0.0));
	out[4] = cpxMul(cpxMul(cpxSub(cpxAdd(cpxAdd(in0, in4), cpxAdd(in1, in5)), cpxAdd(cpxAdd(in0, in4), cpxAdd(in1, in5))), make_cpx(1.000000, -0.000000)), make_cpx(1.000000, 0.0));
	out[5] = cpxMul(cpxMul(cpxSub(cpxAdd(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, -0.707107))), cpxAdd(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, -0.707107)))), make_cpx(1.000000, -0.000000)), make_cpx(1.000000, 0.0));
	out[6] = cpxMul(cpxMul(cpxSub(cpxMul(cpxSub(cpxAdd(in0, in4), cpxAdd(in1, in5)), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(cpxAdd(in0, in4), cpxAdd(in1, in5)), make_cpx(-0.000000, -1.000000))), make_cpx(1.000000, -0.000000)), make_cpx(1.000000, 0.0));
	out[7] = cpxMul(cpxMul(cpxSub(cpxMul(cpxSub(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, -0.707107))), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, -1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, -0.707107))), make_cpx(-0.000000, -1.000000))), make_cpx(1.000000, -0.000000)), make_cpx(1.000000, 0.0));
}

__inline static void fft_x8inv(cpx *in, cpx *out)
{
	cpx in0 = in[0];
	cpx in1 = in[1];
	cpx in2 = in[2];
	cpx in3 = in[3];
	cpx in4 = in[4];
	cpx in5 = in[5];
	cpx in6 = in[6];
	cpx in7 = in[7];
	out[0] = cpxMul(cpxAdd(cpxAdd(cpxAdd(in0, in4), cpxAdd(in1, in5)), cpxAdd(cpxAdd(in0, in4), cpxAdd(in1, in5))), make_cpx(0.125000, 0.0));
	out[1] = cpxMul(cpxAdd(cpxAdd(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, 0.707107))), cpxAdd(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, 0.707107)))), make_cpx(0.125000, 0.0));
	out[2] = cpxMul(cpxAdd(cpxMul(cpxSub(cpxAdd(in0, in4), cpxAdd(in1, in5)), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(cpxAdd(in0, in4), cpxAdd(in1, in5)), make_cpx(-0.000000, 1.000000))), make_cpx(0.125000, 0.0));
	out[3] = cpxMul(cpxAdd(cpxMul(cpxSub(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, 0.707107))), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, 0.707107))), make_cpx(-0.000000, 1.000000))), make_cpx(0.125000, 0.0));
	out[4] = cpxMul(cpxMul(cpxSub(cpxAdd(cpxAdd(in0, in4), cpxAdd(in1, in5)), cpxAdd(cpxAdd(in0, in4), cpxAdd(in1, in5))), make_cpx(1.000000, 0.000000)), make_cpx(0.125000, 0.0));
	out[5] = cpxMul(cpxMul(cpxSub(cpxAdd(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, 0.707107))), cpxAdd(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, 0.707107)))), make_cpx(1.000000, 0.000000)), make_cpx(0.125000, 0.0));
	out[6] = cpxMul(cpxMul(cpxSub(cpxMul(cpxSub(cpxAdd(in0, in4), cpxAdd(in1, in5)), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(cpxAdd(in0, in4), cpxAdd(in1, in5)), make_cpx(-0.000000, 1.000000))), make_cpx(1.000000, 0.000000)), make_cpx(0.125000, 0.0));
	out[7] = cpxMul(cpxMul(cpxSub(cpxMul(cpxSub(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, 0.707107))), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(cpxMul(cpxSub(in2, in6), make_cpx(-0.000000, 1.000000)), cpxMul(cpxSub(in3, in7), make_cpx(-0.707107, 0.707107))), make_cpx(-0.000000, 1.000000))), make_cpx(1.000000, 0.000000)), make_cpx(0.125000, 0.0));
}


#endif