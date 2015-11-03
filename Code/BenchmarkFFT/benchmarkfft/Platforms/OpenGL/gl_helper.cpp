#include "gl_helper.h"
#include "../../Common/mymath.h"

double gl_query_time(GLuint q[NUM_TESTS][2])
{
    double measures[NUM_TESTS];
    GLint stopTimerAvailable = 0;
    while (!stopTimerAvailable)
        glGetQueryObjectiv(q[NUM_TESTS - 1][1], GL_QUERY_RESULT_AVAILABLE, &stopTimerAvailable);
    double best_time = 99999999999.9;
    for (int i = 0; i < NUM_TESTS; ++i) {
        GLuint64 start, stop;
        glGetQueryObjectui64v(q[i][0], GL_QUERY_RESULT, &start);
        glGetQueryObjectui64v(q[i][1], GL_QUERY_RESULT, &stop);
        measures[i] = (stop - start) / 1000.0;
    }
    return average_best(measures, NUM_TESTS);
}

void gl_prepare_file(gl_args* args, LPCWSTR shader_file)
{
    std::string str = get_file_content(shader_file);
    manip_content(&str, L"LOCAL_DIM_X", args->threads.x);
    manip_content(&str, L"SHARED_MEM_SIZE", (args->threads.x << 1));
    set_file_content(shader_file, str);
}

void gl_setup_program(gl_args *a, bool gen_buffers, LPCWSTR shader_file)
{
    gl_prepare_file(a, shader_file);
    GLuint program = glCreateProgram();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    GLint length;
    a->shader_src = get_kernel_src(shader_file, &length);
    glShaderSource(shader, 1, &a->shader_src, &length);
    glCompileShader(shader);
    int rvalue;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue) {
        fprintf(stderr, "Error in compiling the compute shader\n");
        GLchar log[10240];
        GLsizei length;
        glGetShaderInfoLog(shader, 10239, &length, log);
        fprintf(stderr, "Compiler log:\n%s\n", log);
        getchar();
        exit(40);
    }
    glAttachShader(program, shader);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &rvalue);
    if (!rvalue) {
        fprintf(stderr, "Error in linking compute shader program\n");
        GLchar log[10240];
        GLsizei length;
        glGetProgramInfoLog(program, 10239, &length, log);
        fprintf(stderr, "Linker log:\n%s\n", log);
        getchar();
        exit(41);
    }
    if (gen_buffers) {
        GLuint buffers[2];
        glGenBuffers(2, buffers);
        a->buf_in = buffers[0];
        a->buf_out = buffers[1];
    }
    a->program = program;
}

void gl_load_buffer(GLuint buffer, cpx* data, const int binding, const int n)
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(cpx), data ? &data[0] : NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer);
}

void gl_read_buffer(GLuint buffer, cpx** data, const int n)
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    *data = (cpx *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
}

void gl_setup(gl_args* a_local, gl_args* a_global, cpx* in, cpx* out, int group_size, const int n)
{
    LPCWSTR shader_file_local = L"Platforms/OpenGL/gl_cshader_local.glsl";
    LPCWSTR shader_file_global = L"Platforms/OpenGL/gl_cshader_global.glsl";

    // "Local" compute shader
    a_local->groups.x = (n >> 1) > group_size ? ((n >> 1) / group_size) : 1;
    a_local->threads.x = (n >> 1) > group_size ? group_size : n >> 1;
    gl_setup_program(a_local, true, shader_file_local);
    gl_load_buffer(a_local->buf_in, in, 0, n);
    gl_load_buffer(a_local->buf_out, out, 1, n);

    // "Global" compute shader
    memcpy(a_global, a_local, sizeof(gl_args));
    gl_setup_program(a_global, false, shader_file_global);
}

void gl_shakedown(gl_args *a)
{
    if (a->shader_src != NULL)
        free(a->shader_src);
}