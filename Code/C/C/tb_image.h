#ifndef TB_IMAGE_H
#define TB_IMAGE_H

#include "tb_math.h"

void img_to_cpx(unsigned char *img, my_complex **com, uint32_t N);
void cpx_to_img(my_complex **com, unsigned char *img, uint32_t N, unsigned char mag);
void fft_shift(unsigned char *in, unsigned char *out, uint32_t N);

int generate_test_image_set(char *filename, char *groupname, uint32_t size);
int writeppm(char *filename, int width, int height, unsigned char *data);
unsigned char *readppm(char *filename, int *width, int *height);

#endif