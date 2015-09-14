#ifndef GENCODE_h
#define GENCODE_H 

#include <string>
#include <regex>
#include <sstream>
#include <algorithm>
#include <regex>

#include "tb_definitions.h"
#include "genHelper.h"
#include "tb_math.h"

void createFixedSizeFFT(std::string name, const int max_n, const int no_rev, const int writeToFile);

#endif