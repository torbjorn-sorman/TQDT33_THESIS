#include "genCode.h"

typedef void(*algorithmSpec)(int index, int depth, int n, int *lower, int *upper, int *p, int *addPredicate);

// Constant Geometry specs
void specs_const(int index, int depth, int n, int *lower, int *upper, int *p, int *addPredicate)
{
    *lower = index / 2;
    *upper = index / 2 + n / 2;
    *p = ((*lower) & (0xffffffff << (depth - 1)));
    *addPredicate = (index % 2 == 0);
}

//  Cooley and Tukey -style
void specs_reg(int index, int depth, int n, int *lower, int *upper, int *p, int *addPredicate)
{
    int i = index / 2;
    int bit = log2_32(n) - depth;
    int dist = (n >> depth);
    int steps = depth;
    
    unsigned int pmask = (dist - 1) << steps;
    unsigned int lmask = (0xFFFFFFFF << bit);
    *lower = i + (i & lmask);
    *upper = (*lower) + dist;
    *p = ((i << steps) & pmask) >> 1;
    *addPredicate = (index % (2 * dist)) < dist;
}

// Generic generator for 2^k size and Constant and Regular
expr *_gen(fft_direction dir, algorithmSpec algSpecs, int index, int depth, const int n)
{
    // Atom (input value)
    if (depth == 0)
        return make_in_expr(index);

    int in_low = 0;
    int in_high = 0;
    int p = 0;    
    int addP = 0;

    // Select input for this step
    algSpecs(index, depth, n, &in_low, &in_high, &p, &addP);

    std::stringstream fmt;
    fmt << index << " " << depth << " " << in_low << " " << in_high << " " << p << " " << addP;
    printf("%s\n", fmt.str().c_str());

    // Twiddle factor angle
    double angle = ((dir * M_2_PI) * p) / (double)n;

    // Get input for this step
    expr *low = _gen(dir, algSpecs, in_low, depth - 1, n);
    expr *high = _gen(dir, algSpecs, in_high, depth - 1, n);

    // Make expression depending on if low or high (index value)
    if (addP)
        return make_expr(CPX_ADD, low, high);
    else
        return make_expr(CPX_MUL, make_expr(CPX_SUB, low, high), make_cpx_expr(cos(angle), sin(angle)));
}

std::string generate_body(algorithmSpec specs, fft_direction direction, const int n)
{
    int depth = log2_32(n);
    int lead = 32 - depth;
    expr **output = (expr **)malloc(sizeof(expr *) * n);

    double scale = direction == FORWARD_FFT ? 1.0 : 1.0 / ((double)n);
    // Build the algorithm by recursivly traverse the path for each output.
    for (int i = 0; i < n; ++i) {
        output[i] = make_out_expr(BIT_REVERSE(i, lead), _gen(direction, specs, i, depth, n), scale);
    }    

    // Reorder, so that output is sequential
    std::sort(output, output + n, [](expr *a, expr *b){ return(a->index < b->index); });    

    // Write to string
    std::stringstream fmt;    
    
    // Function declaration
    fmt << "__inline static void fft_x" << n << (direction == INVERSE_FFT ? "inv" : "") << "(cpx *in, cpx *out)\n{\n";
        
    int useLocal = 1;
    // If useLocal then all input values are read to local values before operations.
    if (useLocal == 1) {
        for (int i = 0; i < n; ++i) {
            fmt << "\tcpx in" << i << " = in[" << i << "];\n";
        }
    }
    for (int i = 0; i < n; ++i) {
        fmt << "\t" << exprToString(output[i], useLocal).c_str() << ";\n";
    }
    // Close function and return function budy
    fmt << "}\n\n";

    printf("\n");
    return fmt.str();
}

void createFixedSizeFFT(std::string name, const int max_n)
{
    char filename[64];
    sprintf_s(filename, 64, "fft_generated_fixed_%s.h", name.c_str());
    FILE *f;
    fopen_s(&f, filename, "w");

    algorithmSpec specs = (name.compare("const") == 0 ? specs_const : specs_reg);

    // File header
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    fprintf_s(f, "#ifndef FFT_GENERATED_FIXED_%s_H\n", name.c_str());
    fprintf_s(f, "#define FFT_GENERATED_FIXED_%s_H\n\n", name.c_str());
    fprintf_s(f, "#include \"tb_definitions.h\"\n");
    fprintf_s(f, "#include \"tb_fft_helper.h\"\n\n");
    fprintf_s(f, "#define GENERATED_FIXED_%s\n\n", name.c_str());

    // Function declarations and bodies (__inline static void ...)
    for (int i = 4; i <= max_n; i *= 2) {
        fprintf_s(f, "%s", generate_body(specs, FORWARD_FFT, i));
        fprintf_s(f, "%s", generate_body(specs, INVERSE_FFT, i));
    }

    // End and close file.
    fprintf_s(f, "\n#endif");
    fclose(f);
}