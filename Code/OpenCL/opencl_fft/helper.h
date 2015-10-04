#ifndef HELPER_H
#define HELPER_H

#include <Windows.h>
#include <vector>
#include <fstream>
#include <CL\cl.h>

#include "definitions.h"

void startTimer();
double stopTimer();

std::string getKernel(const char *filename);

unsigned int power(const unsigned int base, const unsigned int exp);
unsigned int power2(const unsigned int exp);

int checkError(cpx *seq, cpx *ref, float refScale, const int n, int print);
int checkError(cpx *seq, cpx *ref, const int n, int print);
int checkError(cpx *seq, cpx *ref, const int n);

cpx *get_seq(const int n);
cpx *get_seq(const int n, const int sinus);
cpx *get_seq(const int n, cpx *src);

double avg(double m[], int n);

void write_console(cpx a);
void write_console(cpx *seq, const int n);

// Fast bit-reversal
static int tab32[32] = {
    0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
};

static __inline int log2_32(int value)
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(unsigned int)(value * 0x07C4ACDD) >> 27];
}

static __inline void oclExecute(oclArgs *args)
{
    clEnqueueNDRangeKernel(args->commands, args->kernel, 1, NULL, &args->global_work_size, &args->local_work_size, 0, NULL, NULL);
    clFinish(args->commands);
}

static __inline void swap(cl_mem *a, cl_mem *b)
{
    cl_mem c = *a;
    *a = *b;
    *b = c;
}

int checkErr(cl_int error, char *msg);
int checkErr(cl_int error, cl_int args, char *msg);

cl_int oclSetup(char *kernelName, cpx *dev_in, oclArgs *args);
void oclRelease(cpx *dev_out, oclArgs *args, cl_int *error);

#endif