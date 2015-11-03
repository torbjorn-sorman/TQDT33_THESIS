#pragma once
#ifndef MYFILE_H
#define MYFILE_H

#include <Windows.h>
#include <string>
#include <fstream>

bool dirExists(const std::string& dirName_in);
FILE *getImageFilePointer(char *name, int n, char *type);
FILE *getTextFilePointer(std::string name, std::string *fname);
char *get_kernel_src(const char *filename, int *length);

#endif