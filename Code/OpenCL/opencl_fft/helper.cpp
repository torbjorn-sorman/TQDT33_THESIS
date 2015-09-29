
#include "helper.h"

std::string getKernel(const char *filename)
{
    std::ifstream in(filename);
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());    
    return contents;
}