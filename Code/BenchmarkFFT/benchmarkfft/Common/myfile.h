#ifndef MYFILE_H
#define MYFILE_H

#include <Windows.h>
#include <string>

static bool dirExists(const std::string& dirName_in)
{
    DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
    if (ftyp == INVALID_FILE_ATTRIBUTES)
        return false;  //something is wrong with your path!

    if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
        return true;   // this is a directory!

    return false;    // this is not a directory!
}

static FILE *getImageFilePointer(char *name, int n, char *type)
{
    FILE  *fp;
    std::string dir = "out/img/" + std::string(name);    
    char filename[60];    
    sprintf_s(filename, 60, "%s/%u_%s.ppm", dir.c_str(), n, type);
    if (!dirExists("out"))      CreateDirectory("out", NULL);
    if (!dirExists("out/img"))  CreateDirectory("out/img", NULL);
    if (!dirExists(dir))        CreateDirectory(dir.c_str(), NULL);
    fopen_s(&fp, filename, "wb");
    return fp;
}

static FILE *getTextFilePointer(std::string name, std::string *fname)
{
    FILE  *fp;
#ifdef _WIN64
    std::string dir = "out/x64/";
#else
    std::string dir = "out/Win32/";
#endif
    if (!dirExists("out"))      CreateDirectory("out", NULL);
    if (!dirExists(dir))        CreateDirectory(dir.c_str(), NULL);
    fopen_s(&fp, (dir + name).c_str(), "w");
    *fname = dir + name;
    return fp;
}

#endif