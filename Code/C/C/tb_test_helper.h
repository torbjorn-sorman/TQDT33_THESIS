#ifndef TB_TEST_HELPER_H
#define TB_TEST_HELPER_H

#include "tb_definitions.h"

int checkError(tb_cpx *seq, tb_cpx *ref, const int n, int print);
int checkError(tb_cpx **seq, tb_cpx **ref, const int n, int print);
int cmp(const void *x, const void *y);
int cmp(unsigned char *a, unsigned char *b, const int n);
int equal(tb_cpx a, tb_cpx b);
double avg(double m[], int n);
double abs_diff(tb_cpx a, tb_cpx b);

tb_cpx *get_seq(const int n);
tb_cpx *get_seq(const int n, const int sinus);
tb_cpx *get_seq(const int n, tb_cpx *src);
tb_cpx **get_seq2d(const int n);
tb_cpx **get_seq2d(const int n, const int type);
tb_cpx **get_seq2d(const int n, tb_cpx **src);
void copy_seq2d(tb_cpx **from, tb_cpx **to, const int n);

unsigned char *get_empty_img(const int w, const int h);

void free_seq2d(tb_cpx **seq, const int n);

#endif