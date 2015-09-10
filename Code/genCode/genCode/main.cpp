#include <memory>
#include <iostream>
#include <string>
#include <cstdio>

template<typename ... Args>
string string_format(const std::string& format, Args ... args)
{
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

typedef enum { CPX_NONE, CPX_ADD, CPX_SUB, CPX_MUL, CPX_MAKE } cpx_op_type;

struct expr {
    cpx_op_type op;
    int index;
    struct expr *left;
    struct expr *right;
    cpx *value;
};

expr make_expr(cpx_op_type op, int index, expr *left, expr *right)
{
    return{ op, index, left, right, &make_cpx(0, 0) };
}

expr make_expr(cpx_op_type op, expr *left, expr *right)
{
    return{ op, -1, left, right, &make_cpx(0, 0) };
}

expr make_expr(cpx *value)
{
    return{ CPX_NONE, -1, NULL, NULL, value };
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
        return &make_expr(CPX_MUL, BIT_REVERSE(index, lead), &make_expr(CPX_SUB, low, high), &make_expr(&make_cpx(cos(0.0), sin(0.0))));
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
    case CPX_NONE:
        if (e->value == NULL)
            return string_format("in[%d]", e->index);
        else
            return string_format("%f", e->value);
    default:
        return "";
        break;
    }
}

std::string *toStr(expr *t, int n)
{
    std::string *strs = (std::string *)malloc(sizeof(std::string) * n);
    for (int i = 0; i < n; ++i) {
        strs[i] = _toStr(&t[i]);
    }
    return strs;
}

std::string generate(const int n)
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
            output[i] = make_expr(CPX_MUL, BIT_REVERSE(i, lead), &make_expr(CPX_SUB, low, high), &make_expr(&make_cpx(cos(0.0), sin(0.0))));
    }
    std::string *strs = toStr(output, n);
    std::string out = "";
    for (int i = 0; i < n; ++i) {
        out += ";\n" + strs[i];
    }
    free(output);
    free(strs);
    return out + ";";
}

void generator(const int n)
{
    char filename[64] = "";
    FILE *f;
    fopen_s(&f, "out/generated.txt", "w");
    fprintf_s(f, generate(n).c_str());
    fclose(f);
}