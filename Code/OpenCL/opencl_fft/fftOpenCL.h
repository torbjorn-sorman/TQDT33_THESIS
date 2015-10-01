#ifndef FFTGPUSYNC_H
#define FFTGPUSYNC_H

#include "definitions.h"
#include "CL\cl.h"
#include <stdio.h>
#include "helper.h"

// Syncronize only on GPU after launch (failed to sync over blocks)
int     GPUSync_validate(const int n);
double  GPUSync_performance(const int n);

// Partially sync on GPU after problem size reduced.
int     PartSync_validate(const int n);
double  PartSync_performance(const int n);

#endif