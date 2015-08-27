#ifndef TB_IMAGE_H
#define TB_IMAGE_H

#include "tb_definitions.h"

void img_to_cpx(unsigned char *img, tb_cpx **com, const int n);
void cpx_to_img(tb_cpx **com, unsigned char *img, const int n, unsigned char mag);
void fft_shift(unsigned char *in, unsigned char *out, const int n);

int generate_test_image_set(char *filename, char *groupname, const int size);
int writeppm(char *filename, int width, int height, unsigned char *data);
unsigned char *readppm(char *filename, int *width, int *height);

#endif