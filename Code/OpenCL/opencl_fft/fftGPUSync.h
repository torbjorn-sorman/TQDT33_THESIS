#ifndef FFTGPUSYNC_H
#define FFTGPUSYNC_H

#include "definitions.h"
#include "CL\cl.h"
#include <stdio.h>
#include "helper.h"

int     GPUSync_validate    (const int n);
double  GPUSync_performance (const int n);

#endif