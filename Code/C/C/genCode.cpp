#include <memory>
#include <iostream>
#include <string>
#include <cstdio>

#include "genCode.h"

#include "tb_math.h"
#include "tb_fft_helper.h"

using namespace std; //Don't if you're in a header-file

template<typename ... Args>
string string_format(const std::string& format, Args ... args)
{
    size_t size = sprintf_s(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    unique_ptr<char[]> buf(new char[size]);
    sprintf_s(buf.get(), size, format.c_str(), args ...);
    return string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

typedef enum { CPX_NONE, CPX_REAL, CPX_IMAG, CPX_ADD, CPX_SUB, CPX_MUL, CPX_MAKE } cpx_op_type;

struct expr {
    cpx_op_type op;
    int index;
    struct expr *left;
    struct expr *right;
    double value;
};

expr make_expr(cpx_op_type op, int index, expr *left, expr *right)
{
    return{ op, index, left, right, 0.0 };
}

expr make_expr(cpx_op_type op, expr *left, expr *right)
{
    return{ op, -1, left, right, 0.0 };
}

expr make_expr(double r, double i)
{
    return{ CPX_MAKE, -1, &make_expr(CPX_REAL, r), &make_expr(CPX_REAL, i), 0.0 };
}

expr make_expr(cpx_op_type op, double val)
{
    return{ op, -1, NULL, NULL, val };
}

expr make_expr(int index)
{
    return{ CPX_NONE, index, NULL, NULL, NULL };
}

expr *_gen(int index, int depth, const int lead, const int n)
{
    if (depth == 0) {
        return &make_expr(index);
    }
    expr *low = _gen(index / 2, depth - 1, lead, n);
    expr *high = _gen(index / 2 + n / 2, depth - 1, lead, n);
    if (index % 2 == 0)
        return &make_expr(CPX_ADD, BIT_REVERSE(index, lead), low, high);
    else
        return &make_expr(CPX_MUL, BIT_REVERSE(index, lead), &make_expr(CPX_SUB, low, high), &make_expr(cos(0.0), sin(0.0)));
}

std::string _toStr(expr *e)
{
    int l1, l2;
    switch (e->op)
    {
    case CPX_ADD:
        return string_format("cpxAdd(%s, %s)", _toStr(e->left), _toStr(e->right));
    case CPX_SUB:
        return string_format("cpxSub(%s, %s)", _toStr(e->left), _toStr(e->right));
    case CPX_MUL:
        return string_format("cpxMul(%s, %s)", _toStr(e->left), _toStr(e->right));
    case CPX_MAKE:
        return string_format("make_cpx(%s, %s)", _toStr(e->left), _toStr(e->right));
    case CPX_REAL:
    case CPX_IMAG:
        return string_format("%f", e->value);
    case CPX_NONE:
        return string_format("in[%d]", e->index);
    default:
        return "";
        break;
    }
}

std::string *toStr(expr *t, int n)
{
    string *strs = new string[n]();
    for (int i = 0; i < n; ++i) {
        strs[i] = _toStr(&t[i]);
    }
    return strs;
}

void generate(const int n)
{
    int depth = log2_32(n);
    int lead = 32 - depth;
    int n2 = n / 2;
    expr *output = (expr *)malloc(sizeof(expr) * n);
    for (int i = 0; i < n; ++i) {        
        expr *low = _gen(i / 2, depth - 1, lead, n);
        expr *high = _gen(i / 2 + n2, depth - 1, lead, n);
        if (i % 2 == 0)
            output[i] = make_expr(CPX_ADD, BIT_REVERSE(i, lead), low, high);
        else
            output[i] = make_expr(CPX_MUL, BIT_REVERSE(i, lead), &make_expr(CPX_SUB, low, high), &make_expr(cos(0.0), sin(0.0)));
    }        


    char filename[64] = "";
    FILE *f;
    fopen_s(&f, "out/generated.txt", "w");
    for (int i = 0; i < n; ++i) {        
        fprintf_s(f, "%s;\n", _toStr(&output[i]));
    }
    fclose(f);
    free(output);    
}

void generator(const int n)
{
    generate(n);
}