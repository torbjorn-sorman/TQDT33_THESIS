#pragma once
#ifndef _IMGLIB_0
#define _IMGLIB_0

#include <stdio.h>
#include "../Definitions.h"
#include "myfile.h"

typedef unsigned char color_component;
typedef color_component pixel[3];
typedef struct {
    unsigned int width;
    unsigned int height;
    pixel * buf;
} image_t;
typedef image_t * image;

image alloc_img(unsigned int width, unsigned int height);
void free_img(image);
void fill_img(image img,
    color_component r,
    color_component g,
    color_component b);
void put_pixel_unsafe(
    image img,
    unsigned int x,
    unsigned int y,
    color_component r,
    color_component g,
    color_component b);
void put_pixel_clip(
    image img,
    unsigned int x,
    unsigned int y,
    color_component r,
    color_component g,
    color_component b);
#define GET_PIXEL(IMG, X, Y) (IMG->buf[ ((Y) * IMG->width + (X)) ])

image get_ppm(FILE *pf);
void output_ppm(FILE *fd, image img);

//
// Additions to handle my custom format <-> ppm-format
//

void read_image(cpx *dst, char *name, int *n);
void write_image(char *name, char *type, cpx* seq, int n);
void write_image(char *name, char *type, cpx** seq, int n);
void write_normalized_image(char *name, char *type, cpx* seq, int n, bool doFFTShift);
void write_normalized_image(char *name, char *type, cpx** seq, int n, bool doFFTShift);
void normalized_image(cpx* seq, int n);
void fftShift(cpx *dst, cpx *src, int n);
void clear_image(cpx* seq, int n);

#endif
