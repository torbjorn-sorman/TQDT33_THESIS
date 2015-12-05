#include "Definitions.h"

int vendor_gpu = VENDOR_NVIDIA;
int number_of_tests = 8;
#if defined(_NVIDIA)
int batch_total_points = 1048576;//67108864;
#elif defined(_AMD)
int batch_total_points = 67108864;
#endif