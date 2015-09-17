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
    else
        return make_expr(CPX_MUL, make_expr(CPX_SUB, low, high), make_cpx_expr(cos(angle), sin(angle)));
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
    for (int i = 0; i < str.length() - 2; ++i) {
        if (str[i] == '(' && str[i + 1] == '(') {
            cnt = 1;
            for (int j = i + 2; j < str.length() - 1; ++j) {
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
    while (std::regex_search(clean_str.c_str(), cm, std::regex("(\\(\\(.*\\) \\* \\(-?0\\.0{6}\\)\\))"))) {
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
    while (std::regex_search(clean_str.c_str(), cm, std::regex("( \\* ((\\(1\\.0{6}\\))|(1\\.0{6}))\\))"))) {
        start = cm.position();
        end = cm.length() - 1;
        clean_str.erase(start, end);
    }
    // Alter sign when mul with -1.000000     
    while (std::regex_search(clean_str.c_str(), cm, std::regex("(\\(\\(.*\\) \\* \\(-1\\.0{6}\\)\\))"))) {
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
        clean_str.replace(start, end, (" "+sign+" " + clean_str.substr(rs, re)).c_str());
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

std::string generate_body(algorithmSpec specs, fft_direction direction, const int no_rev, const int n)
{
    int depth = log2_32(n);
    int lead = 32 - depth;
    expr **output = (expr **)malloc(sizeof(expr *) * n);

    double scale = ((direction == FORWARD_FFT) || no_rev) ? 1.0 : 1.0 / ((double)n);
    // Build the algorithm by recursivly traverse the path for each output.
    for (int i = 0; i < n; ++i) {
        if (no_rev)
            output[i] = make_out_expr(i, _gen(direction, specs, i, depth, n), scale);
        else
            output[i] = make_out_expr(BIT_REVERSE(i, lead), _gen(direction, specs, i, depth, n), scale);
    }    

    // Reorder, so that output is sequential
    std::sort(output, output + n, [](expr *a, expr *b){ return(a->index < b->index); });    

    // Write to string
    std::stringstream fmt;    
    
    // Function declaration
    fmt << "__inline static void fft_x" << n << (direction == INVERSE_FFT ? "inv" : "") << "(cpx *in, cpx *out)\n{\n";
    
    // If useLocal then all input values are read to local values before operations.        
    for (int i = 0; i < n; ++i) {
        fmt << "\tcpx in" << i << " = in[" << i << "];\n";
    }
    for (int i = 0; i < n; ++i) {
        fmt << "\t" << cleanup(exprToString(output[i], 1, CPX_REAL)).c_str() << ";\n";
        fmt << "\t" << cleanup(exprToString(output[i], 1, CPX_IMAG)).c_str() << ";\n";
        //fmt << "\t" << exprToString(output[i], 1).c_str() << ";\n";        
    }
    // Close function and return function budy
    fmt << "}\n\n";    
    return fmt.str();
}

void createFixedSizeFFT(std::string name, const int no_rev, const int max_n, int writeToFile)
{
    algorithmSpec specs = (name.compare("const") == 0 ? specs_const : specs_reg);

    if (writeToFile) {
        char filename[64];
        sprintf_s(filename, 64, "fft_generated_fixed_%s.h", name.c_str());
        FILE *f;
        fopen_s(&f, filename, "w");

        // File header
        std::transform(name.begin(), name.end(), name.begin(), ::toupper);
        fprintf_s(f, "#ifndef FFT_GENERATED_FIXED_%s_H\n", name.c_str());
        fprintf_s(f, "#define FFT_GENERATED_FIXED_%s_H\n\n", name.c_str());
        fprintf_s(f, "#include \"tb_definitions.h\"\n");
        fprintf_s(f, "#include \"tb_fft_helper.h\"\n\n");    

        // Function declarations and bodies (__inline static void ...)
        std::stringstream stream;
        for (int i = 4; i <= max_n; i *= 2) {
            printf("Generating %d ...", i);
            stream << "#define GENERATED_FIXED_"<< i << "\n\n";
            stream << generate_body(specs, FORWARD_FFT, no_rev, i).c_str() << "\n";
            stream << generate_body(specs, INVERSE_FFT, no_rev, i).c_str() << "\n";
            printf("done.\n");
        }
        fprintf_s(f, stream.str().c_str());

        // End and close file.
        fprintf_s(f, "\n#endif");
        fclose(f);
    }
    else {
        for (int i = 4; i <= max_n; i *= 2) {
            printf("Generating %d ...", i);            
            generate_body(specs, FORWARD_FFT, no_rev, i).c_str();
            generate_body(specs, INVERSE_FFT, no_rev, i).c_str();
            printf("done.\n");
        }
    }
}