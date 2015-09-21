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
    int bit = depth - 1;
    int dist = (n >> depth);
    *lower = (index % dist) + ((index >> (log2_32(n) - bit)) * dist * 2);
    *upper = (*lower) + dist;
    *p = ((index % dist) << bit) & (0xFFFFFFFF << bit);
    *addPredicate = ((*lower) == index);
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

    // Twiddle factor angle
    double angle = ((dir * M_2_PI) * p) / (double)n;

    // Get input for this step
    expr *low = _gen(dir, algSpecs, in_low, depth - 1, n);
    expr *high = _gen(dir, algSpecs, in_high, depth - 1, n);

    // Make expression depending on if low or high (index value)
    if (addP)
        return make_expr(CPX_ADD, low, high);
    else {
        return make_expr(CPX_MUL, make_expr(CPX_SUB, low, high), make_cpx_expr(cos(angle), sin(angle)));
    }
}

int find_start(std::string str)
{
    int match = 1;
    for (int i = str.length() - 2; i >= 0; --i) {
        if (str[i] == ')')
            ++match;
        else if (str[i] == '(')
            --match;
        if (match == 0) {
            return i;
        }
    }
    return 0;
}

int find_redundant_parenthesis(std::string str, int *start, int *end)
{
    int cnt;
    for (int i = 0; i < (int)str.length() - 2; ++i) {
        if (str[i] == '(' && str[i + 1] == '(') {
            cnt = 1;
            for (int j = i + 2; j < (int)str.length() - 1; ++j) {
                if (str[j] == '(') ++cnt;
                if (str[j] == ')') {
                    --cnt;
                    if (cnt == 0 && str[j + 1] == ')') {
                        *start = i;
                        *end = j + 1;
                        return 1;
                    }
                }
                if (cnt < 1) break;
            }
        }
    }
    return 0;
}

// Remove uneccessary information
std::string cleanup(std::string str)
{
    std::string clean_str = str;
    std::cmatch cm;
    int start, end, rs, re;
    // Remove statements mul with zero             
    while (std::regex_search(clean_str.c_str(), cm, std::regex("(\\(\\(.*\\) \\* \\(-?0\\.0{6}f\\)\\))"))) {
        start = find_start(cm.str()) + cm.position();
        end = cm.length() + cm.position() - start;
        clean_str.erase(start, end);
    }
    // Remove leading add/sub        
    while (std::regex_search(clean_str.c_str(), cm, std::regex("( [-+] )\\)"))) {
        start = cm.position();
        end = cm.length() - 1;
        clean_str.erase(start, end);
    }
    // Remove mul 1.000000   
    while (std::regex_search(clean_str.c_str(), cm, std::regex("( \\* ((\\(1\\.0{6}f\\))|(1\\.0{6}f))\\))"))) {
        start = cm.position();
        end = cm.length() - 1;
        clean_str.erase(start, end);
    }
    // Alter sign when mul with -1.000000     
    while (std::regex_search(clean_str.c_str(), cm, std::regex("(\\(\\(.*\\) \\* \\(-1\\.0{6}f\\)\\))"))) {
        start = find_start(cm.str()) + cm.position();
        end = cm.length() + cm.position() - start;
        rs = start + 1;
        re = end - 16;
        clean_str.replace(start, end, ("( - " + clean_str.substr(rs, re) + ")").c_str());
    }
    // Remove multiple negative marks    
    while (std::regex_search(clean_str.c_str(), cm, std::regex("( ([-+]+) \\(* - )"))) {
        start = cm.position();
        end = cm.length();
        rs = start + 3;
        re = end - 6;
        std::string sign = (cm[2].compare("-") == 0 ? "+" : "-");
        clean_str.replace(start, end, (" " + sign + " " + clean_str.substr(rs, re)).c_str());
    }
    // Remove empty add            
    while (std::regex_search(clean_str.c_str(), cm, std::regex("(\\( \\+ \\()"))) {
        start = cm.position() + 1;
        end = cm.length() - 2;
        clean_str.erase(start, end);
    }
    // Remove redundant parenthesis     
    int s, e;
    while (find_redundant_parenthesis(clean_str, &s, &e)) {
        clean_str.erase(s, 1);
        clean_str.erase(e - 1, 1);
    }
    return clean_str;
}

void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty())
        return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}

void clrStream(std::stringstream &s1, std::stringstream &s2)
{
    s1.str("");
    s1.clear();
    s2.str("");
    s2.clear();
}

void progress(int d, int len)
{
    for (int i = 0; i < d; ++i)
        printf("  ");
    printf("|");
    for (int i = 0; i < len; ++i)
        printf("-");
    printf("|\n");
    for (int i = 0; i < d; ++i)
        printf("  ");
    printf("|");
}

std::string fnName(fft_direction d, gen_flag bitOrder, const int l)
{
    std::stringstream fmt;
    fmt << "fft_" << (d == INVERSE_FFT ? "i" : "f") << "_" << (bitOrder == GEN_NORMAL_ORDER ? "n" : "b");
    for (int i = 0; i < 3 - log10(l); ++i)
        fmt << "_";
    fmt << l;
    return fmt.str();
}

std::string generate_body(algorithmSpec specs, fft_direction direction, gen_flag bit_order, const int n)
{
    int depth = log2_32(n);
    int lead = 32 - depth;
    expr **output = (expr **)malloc(sizeof(expr *) * n);
    std::string *outStr = (std::string *)malloc(sizeof(std::string) * n * 2);
    double scale = ((direction == FORWARD_FFT) || bit_order == GEN_NORMAL_ORDER) ? 1.0 : 1.0 / ((double)n);
    // Build the algorithm by recursivly traverse the path for each output.
    for (int i = 0; i < n; ++i) {
        //if (bit_order == GEN_NORMAL_ORDER)
            output[i] = make_out_expr(i, _gen(direction, specs, i, depth, n), 1.0);//scale);
        //else
            //output[i] = make_out_expr(BIT_REVERSE(i, lead), _gen(direction, specs, i, depth, n), scale);
    }

    progress(2, n);
    for (int i = 0; i < n; ++i) {
        printf(".");
        outStr[i*2] = cleanup(exprToString(output[i], 1, CPX_REAL));
        outStr[i*2 + 1] = cleanup(exprToString(output[i], 1, CPX_IMAG));     
    }
    printf("|\n");

    // Reorder, so that output is sequential
    //std::sort(output, output + n, [](expr *a, expr *b){ return(a->index < b->index); });
    std::sort(outStr, outStr + n * 2, [](std::string a, std::string b){ 
        printf("a: %s\n", a.c_str());
        return std::stoi(a.substr(2, a.find("]"))) < std::stoi(b.substr(2, a.find("]")));
    });

    // Write to string
    std::stringstream fmt;

    // Function declaration

    fmt << "__inline static void ";
    fmt << fnName(direction, bit_order, n);
    fmt << "(cpx *i, cpx *o" << ")\n{\n";

    // If useLocal then all input values are read to local values before operations.        
    for (int i = 0; i < n; ++i) {
        fmt << "\tcpx i" << i << " = i[" << i << "];\n";
    }   
    for (int i = 0; i < n; ++i) {
        fmt << "\t" << outStr[i*2].c_str() << ";\n";
        fmt << "\t" << outStr[i*2 + 1].c_str() << ";\n";
        //fmt << "\t" << exprToString(output[i], 1).c_str() << ";\n";        
    }
    // Close function and return function budy
    fmt << "}\n\n";
    return fmt.str();
}

std::string genCaseIndex(int index)
{
    std::stringstream fmt;
    fmt << "case ";
    int len = (index == 10 ? 2 : (index < 10 ? 3 : 4 - log10(index)));
    for (int i = 0; i < len; ++i) {
        fmt << " ";
    }
    fmt << index <<": ";
    return fmt.str();
}

std::string createFn(const int n) {
    std::stringstream strm;
    strm << "__inline static int fixed_size_fft(fft_direction dir, cpx *in, cpx *out, gen_flag bitOrder, const int n)\n";
    strm << "{\n";    
    int cnt = log2_32(n) - 2;
    strm << "\tif(n > " << n << ")\n\t\treturn 0;\n";
    strm << "\tint index = (bitOrder == GEN_NORMAL_ORDER) * " << (cnt + 1) * 2;
    strm << " + (dir == INVERSE_FFT) * " << (cnt + 1);
    strm << " + (log2_32(n) - 2);\n";
    strm << "\tswitch (index)\n\t{\n";
    int index = 0;
    std::string params = "(in, out);\treturn 1;\n";
    for (int i = 4; i <= n; i *= 2) {
        strm << "\t" << genCaseIndex(index++) << fnName(FORWARD_FFT, GEN_BIT_REVERSE_ORDER, i) << params;
    }
    for (int i = 4; i <= n; i *= 2) {
        strm << "\t" << genCaseIndex(index++) << fnName(INVERSE_FFT, GEN_BIT_REVERSE_ORDER, i) << params;
    }
    for (int i = 4; i <= n; i *= 2) {
        strm << "\t" << genCaseIndex(index++) << fnName(FORWARD_FFT, GEN_NORMAL_ORDER, i) << params;
    }
    for (int i = 4; i <= n; i *= 2) {
        strm << "\t" << genCaseIndex(index++) << fnName(INVERSE_FFT, GEN_NORMAL_ORDER, i) << params;
    }
    strm << "\tdefault:\t\t\t\t\t\t\treturn 0;\n\t}\n}\n";
    return strm.str();
}

LARGE_INTEGER st, et, em, fq;

#define QPF QueryPerformanceFrequency
#define QPC QueryPerformanceCounter

#define START_TIME QPF(&fq); QPC(&st)
#define STOP_TIME(RES) QPC(&et); em.QuadPart = et.QuadPart - st.QuadPart; em.QuadPart *= 1000000; em.QuadPart /= fq.QuadPart;(RES) = (double)em.QuadPart

void createFixedSizeFFT(std::string name, const int max_n, gen_flag file_flag)
{
    double t = 0.0;
    algorithmSpec specs = (name.compare("const") == 0 ? specs_const : specs_reg);
    START_TIME;
    if (file_flag == GEN_TO_FILE) {
        char filename[64];
        sprintf_s(filename, 64, "fft_generated_fixed_%s.h", name.c_str());
        FILE *f;
        fopen_s(&f, filename, "w");

        // File header
        std::transform(name.begin(), name.end(), name.begin(), ::toupper);
        fprintf_s(f, "#ifndef FFT_GENERATED_FIXED_%s_H\n", name.c_str());
        fprintf_s(f, "#define FFT_GENERATED_FIXED_%s_H\n\n", name.c_str());
        fprintf_s(f, "#include \"tb_definitions.h\"\n");
        fprintf_s(f, "#include \"genCode.h\"\n");
        fprintf_s(f, "#include \"tb_fft_helper.h\"\n\n");
        fprintf_s(f, "#define GENERATED_FIXED_SIZE\n\n");

        // Function declarations and bodies (__inline static void ...)
        std::stringstream stream;
        for (int i = 4; i <= max_n; i *= 2) {
            printf("Generating Size %d:\n", i);
            printf("  Forward Bit Reversed:\n");
            stream << generate_body(specs, FORWARD_FFT, GEN_BIT_REVERSE_ORDER, i).c_str() << "\n";
            printf("  Inverse Bit Reversed:\n");
            stream << generate_body(specs, INVERSE_FFT, GEN_BIT_REVERSE_ORDER, i).c_str() << "\n";
            printf("  Forward Normal Order:\n");
            stream << generate_body(specs, FORWARD_FFT, GEN_NORMAL_ORDER, i).c_str() << "\n";
            printf("  Inverse Normal Order:\n");
            stream << generate_body(specs, INVERSE_FFT, GEN_NORMAL_ORDER, i).c_str() << "\n";
            printf("done.\n");
        }
        fprintf_s(f, stream.str().c_str());
        fprintf_s(f, createFn(max_n).c_str());

        STOP_TIME(t);
        fprintf_s(f, "\n// Code generated in %.2f seconds\n", t / 1000000.0);

        // End and close file.
        fprintf_s(f, "\n#endif");
        fclose(f);
    }
    else {
        for (int i = 4; i <= max_n; i *= 2) {
            printf("Generating %d ...", i);
            generate_body(specs, FORWARD_FFT, GEN_BIT_REVERSE_ORDER, i);
            generate_body(specs, INVERSE_FFT, GEN_BIT_REVERSE_ORDER, i);
            generate_body(specs, FORWARD_FFT, GEN_NORMAL_ORDER, i);
            generate_body(specs, INVERSE_FFT, GEN_NORMAL_ORDER, i);
            printf("done.\n");
        }
        STOP_TIME(t);
    }
    printf("Time elapsed:\t%0.2fs\n\n", t / 1000000.0);
}