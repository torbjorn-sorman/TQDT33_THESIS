#ifndef FFT_GENHELPER_H
#define FFT_GENHELPER_H

#include <string>
#include <regex>
#include <sstream>
#include <algorithm>

#include "tb_definitions.h"

typedef enum {
    CPX_NONE,
    CPX_IN,
    CPX_OUT,
    CPX_REAL,
    CPX_IMAG,
    CPX_ADD,
    CPX_SUB,
    CPX_MUL,
    CPX_MAKE
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
    switch (e->op)
    {
    case CPX_ADD:         
        fmt << "(" << left << " + " << right << ")";
        break;
    case CPX_SUB:
        fmt << "(" << left << " - " << right << ")";
        break;
    case CPX_MUL:
        fmt << "((" << left << " * " << right << ") - (" << left << " * " << right << "))";
        break;
    case CPX_MAKE: 
        fmt << "(" << (reImg == CPX_REAL ? left : right) << ")";
        break;
    case CPX_REAL: fmt << e->value;
        break;
    case CPX_IMAG: fmt << e->value;
        break;
    case CPX_IN:
        if (useLocal == 0)
            fmt << "in[" << e->index << "]." << part;
        else
            fmt << "in" << e->index << "." << part;
        break;
    case CPX_OUT: 
        fmt << "out[" << e->index << "]." << part << " = (" << left << " * " << e->value << ")";
        break;
    default: fmt << "";
        break;
    }
    return fmt.str();
}

#endif