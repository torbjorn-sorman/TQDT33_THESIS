#ifndef TB_IMAGE_H
#define TB_IMAGE_H

#include "tb_definitions.h"

void img_to_cpx(unsigned char *img, cpx **com, const int n);
void cpx_to_img(cpx **com, unsigned char *img, const int n, unsigned char mag);
void fft_shift(unsigned char *in, unsigned char *out, const int n);

int generate_test_image_set(char *filename, char *groupname, const int size);
void writeppm(char *filename, const int width, const int height, unsigned char *img);
unsigned char *readppm(char *filename, int *width, int *height);

#endif