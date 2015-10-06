#ifndef GETOPT_H
#define GETOPT_H

#include <stdlib.h>
#include <string>

#include "definitions.h"
#include "tsHelper.cuh"

int parseArguments(benchmarkArgument *arg, int argc, const char* argv[]);

#endif