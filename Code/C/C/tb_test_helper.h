#ifndef TB_TEST_HELPER_H
#define TB_TEST_HELPER_H

#include "tb_definitions.h"

int checkError(cpx *seq, cpx *ref, const int n, int print);
int checkError(cpx **seq, cpx **ref, const int n, int print);
int cmp(const void *x, const void *y);
int cmp(unsigned char *a, unsigned char *b, const int n);
int equal(cpx a, cpx b);
double avg(double m[], int n);
double abs_diff(cpx a, cpx b);

cpx *get_seq(const int n);
cpx *get_seq(const int n, const int sinus);
cpx *get_seq(const int n, cpx *src);
cpx **get_seq2d(const int n);
cpx **get_seq2d(const int n, const int type);
cpx **get_seq2d(const int n, cpx **src);
void copy_seq2d(cpx **from, cpx **to, const int n);

unsigned char *get_empty_img(const int w, const int h);

void free_seq2d(cpx **seq, const int n);

#endif