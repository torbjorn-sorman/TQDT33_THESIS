#ifndef GENCODE_H
#define GENCODE_H 

#include <string>
#include <regex>
#include <sstream>
#include <algorithm>
#include <regex>
#include <Windows.h>

#include "tb_definitions.h"
#include "tb_math.h"

typedef enum {
    GEN_NONE,
    GEN_BIT_REVERSE_ORDER,
    GEN_NORMAL_ORDER,
    GEN_TWIDDLE,
    GEN_WITH_VARIABLE_TWIDDLE,
    GEN_TO_FILE,
    GEN_NO_FILE
} gen_flag;

typedef enum {
    CPX_NONE,
    CPX_IN,
    CPX_OUT,
    CPX_REAL,
    CPX_IMAG,
    CPX_TWREAL,
    CPX_TWIMAG,
    CPX_ADD,
    CPX_SUB,
    CPX_MUL,
    CPX_MAKE,
    CPX_TWIDDLE
} cpx_op_type;

struct expr {
    cpx_op_type op;
    int index;
    struct expr *left;
    struct expr *right;
    double value;
};

__inline static expr *mkExpr(cpx_op_type o, int i, struct expr *l, struct expr *r, double v)
{
    expr *e = (expr *)malloc(sizeof(expr));
    e->op = o;
    e->index = i;
    e->left = l;
    e->right = r;
    e->value = v;
    return e;
}

__inline static expr* make_float_expr(cpx_op_type op, double val)
{
    return mkExpr(op, -1, NULL, NULL, val);
}

__inline static expr* make_index_expr(cpx_op_type op, int p)
{
    return mkExpr(op, p, NULL, NULL, 0.0);
}

__inline static expr* make_cpx_expr(double r, double i)
{
    return mkExpr(CPX_MAKE, -1, make_float_expr(CPX_REAL, r), make_float_expr(CPX_IMAG, i), 0.0);
}

__inline static expr *make_expr(cpx_op_type op, expr *left, expr *right)
{
    return mkExpr(op, -1, left, right, 0.0);
}

__inline static expr *make_in_expr(int index)
{
    return mkExpr(CPX_IN, index, NULL, NULL, NULL);
}

__inline static expr *make_out_expr(int index, expr *e, double scale)
{
    return mkExpr(CPX_OUT, index, e, NULL, scale);
}

__inline static expr* make_twiddle_expr(int p)
{
    return mkExpr(CPX_MAKE, -1, make_index_expr(CPX_TWREAL, p), make_index_expr(CPX_TWIMAG, p), 0.0);
}

// converts expression tree to string
__inline static std::string exprToString(expr *e, int useLocal, cpx_op_type reImg)
{
    std::stringstream fmt;
    fmt << std::fixed;
    std::string part = (reImg == CPX_REAL) ? "r" : "i";
    std::string left, right;
    if (e != NULL) {
        if (e->left != NULL)
            left = exprToString(e->left, useLocal, reImg);
        if (e->right != NULL)
            right = exprToString(e->right, useLocal, reImg);
    }
    else
        return "";
    switch (e->op)
    {
    case CPX_ADD:
        fmt << "(" << left << " + " << right << ")";
        break;
    case CPX_SUB:
        fmt << "(" << left << " - " << right << ")";
        break;
    case CPX_MUL:
        if (reImg == CPX_REAL) {
            std::string left_im = exprToString(e->left, useLocal, CPX_IMAG);
            std::string right_im = exprToString(e->right, useLocal, CPX_IMAG);
            fmt << "((" << left << " * " << right << ") - (" << left_im << " * " << right_im << "))";
        }
        else {
            std::string left_re = exprToString(e->left, useLocal, CPX_REAL);
            std::string right_re = exprToString(e->right, useLocal, CPX_REAL);
            fmt << "((" << left_re << " * " << right << ") + (" << left << " * " << right_re << "))";
        }
        break;
    case CPX_MAKE:
        fmt << "(" << (reImg == CPX_REAL ? left : right) << ")";
        break;
    case CPX_REAL: fmt << e->value << "f";
        break;
    case CPX_IMAG: fmt << e->value << "f";
        break;
    case CPX_TWREAL: fmt << "w" << e->index << ".r";
        break;
    case CPX_TWIMAG: fmt << "w" << e->index << ".i";
        break;
    case CPX_IN:
        if (useLocal == 0)
            fmt << "i[" << e->index << "]." << part;
        else
            fmt << "i" << e->index << "." << part;
        break;
    case CPX_OUT:
        if (e->value == 1.0)
            fmt << "o[" << e->index << "]." << part << " = " << left;
        else
            fmt << "o[" << e->index << "]." << part << " = " << left << " * " << e->value << "f";
        break;
    default: fmt << "";
        break;
    }
    return fmt.str();
}

// converts expression tree to string
__inline static std::string exprToString(expr *e, int useLocal)
{
    std::stringstream fmt;
    fmt << std::fixed;
    std::string left, right;
    if (e != NULL) {
        if (e->left != NULL)
            left = exprToString(e->left, useLocal);
        if (e->right != NULL)
            right = exprToString(e->right, useLocal);
    }
    switch (e->op)
    {
    case CPX_ADD: fmt << "cpxAdd(" << left << ", " << right << ")";
        break;
    case CPX_SUB: fmt << "cpxSub(" << left << ", " << right << ")";
        break;
    case CPX_MUL: fmt << "cpxMul(" << left << ", " << right << ")";
        break;
    case CPX_MAKE: fmt << "make_cpx(" << left << ", " << right << ")";
        break;
    case CPX_TWIDDLE: fmt << "w" << e->index;
        break;
    case CPX_REAL: fmt << e->value;
        break;
    case CPX_IMAG: fmt << e->value;
        break;
    case CPX_IN:
        if (useLocal == 0)
            fmt << "i[" << e->index << "]";
        else
            fmt << "i" << e->index;
        break;
    case CPX_OUT: fmt << "o[" << e->index << "] = cpxMul(" << left << ", make_cpx(" << e->value << ", 0.0))";
        break;
    default: fmt << "";
        break;
    }
    return fmt.str();
}

void createFixedSizeFFT(std::string name, const int max_n, gen_flag file_flag);

#endif