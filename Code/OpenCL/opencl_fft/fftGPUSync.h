#ifndef FFTGPUSYNC_H
#define FFTGPUSYNC_H

#include "definitions.h"
#include "CL/cl.h"
#include <stdio.h>

int     GPUSync_validate    (cInt n);
double  GPUSync_performance (cInt n);

#endif