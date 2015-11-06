#include "myfile.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>

bool dirExists(const std::string& dirName_in)
{
    DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
    if (ftyp == INVALID_FILE_ATTRIBUTES)
        return false;  //something is wrong with your path!

    if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
        return true;   // this is a directory!

    return false;    // this is not a directory!
}

FILE *get_img_file_pntr(char *name, int n, char *type)
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

FILE *get_txt_file_pntr(std::string name, std::string *fname)
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

char *get_kernel_src(LPCWSTR file_name, int *length)
{
    std::ifstream in(file_name);
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t len = contents.size() + 1;
    char *src = (char *)malloc(sizeof(char) * len);
    strcpy_s(src, sizeof(char) * len, contents.c_str());
    src[contents.size()] = '\0';
    if (length)
        *length = int(len);
    return src;
}

char *get_kernel_src(std::string file_name, int *length)
{
    std::ifstream in(file_name);
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t len = contents.size() + 1;
    char *src = (char *)malloc(sizeof(char) * len);
    strcpy_s(src, sizeof(char) * len, contents.c_str());
    src[contents.size()] = '\0';
    if (length)
        *length = int(len);
    return src;
}

char *get_kernel_src_from_string(std::string contents, int *length)
{
    size_t len = contents.size() + 1;
    char *src = (char *)malloc(sizeof(char) * len);
    strcpy_s(src, sizeof(char) * len, contents.c_str());
    src[contents.size()] = '\0';
    if (length)
        *length = int(len);
    return src;
}

std::string get_file_content(LPCWSTR shader_file)
{
    std::ifstream in_file(shader_file);
    std::stringstream buffer;
    buffer << in_file.rdbuf();
    std::string content = buffer.str();
    in_file.close();
    return content;
}

#include "atlstr.h"

void manip_content(std::string* content, LPCWSTR var_name, int value)
{
    std::stringstream reg_expr;
    reg_expr << "(#define\\s*" << CW2A(var_name) << ")\\s*\\d*";
    std::regex e(reg_expr.str());
    std::stringstream fmt;
    fmt << "$1 " << value;
    *content = std::regex_replace(*content, e, fmt.str());
}

void set_file_content(LPCWSTR shader_file, std::string content)
{
    std::ofstream out_file(shader_file);
    out_file << content;
    out_file.close();
}