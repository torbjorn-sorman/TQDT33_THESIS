#include "gl_helper.h"
#include "../../Common/mathutil.h"

double gl_query_time(GLuint q[][2])
{
    double measures[64];
    GLint stopTimerAvailable = 0;
    while (!stopTimerAvailable)
        glGetQueryObjectiv(q[number_of_tests - 1][1], GL_QUERY_RESULT_AVAILABLE, &stopTimerAvailable);
    double best_time = 99999999999.9;
    for (int i = 0; i < number_of_tests; ++i) {
        GLuint64 start, stop;
        glGetQueryObjectui64v(q[i][0], GL_QUERY_RESULT, &start);
        glGetQueryObjectui64v(q[i][1], GL_QUERY_RESULT, &stop);
        measures[i] = (stop - start) / 1000.0;
        glDeleteQueries(2, q[i]);
    }
    return average_best(measures, number_of_tests);
}

char* gl_prepare_source(gl_args* args, LPCWSTR shader_file, int* length, bool transpose_shader)
{
    std::string str = get_file_content(shader_file);
    if (transpose_shader) {
        manip_content(&str, L"WIDTH", (args->groups.y * args->threads.x) << 1);
        manip_content(&str, L"TILE_DIM", args->tile_dim);
    } else {
        manip_content(&str, L"LOCAL_DIM_X", args->threads.x);
        manip_content(&str, L"SHARED_MEM_SIZE", (args->threads.x << 1));
    }
    return get_kernel_src_from_string(str, length);
}

void gl_log(GLuint shader, const char* type, const char* msg)
{
    fprintf(stderr, msg);
    GLchar log[10240];
    GLsizei length;
    glGetShaderInfoLog(shader, 10239, &length, log);
    fprintf(stderr, "%s log:\n%s\n", type, log);
#pragma warning(suppress: 6031)
    getchar();
    exit(40);
}

void gl_setup_program(gl_args *arg, bool gen_buffers, LPCWSTR shader_file, bool transpose_shader)
{
    
    GLuint program = glCreateProgram();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    GLint length;
    arg->shader_src = gl_prepare_source(arg, shader_file, &length, transpose_shader);
    glShaderSource(shader, 1, &arg->shader_src, &length);
    glCompileShader(shader);
    int rvalue;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue) {
        gl_log(shader, "Compiler", "Error in compiling the compute shader\n");
    }
    glAttachShader(program, shader);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &rvalue);
    if (!rvalue) {
        gl_log(shader, "Linker", "Error in linking compute shader program\n");
    }
    if (gen_buffers) {
        GLuint buffers[2];
        glGenBuffers(2, buffers);
        arg->buf_in = buffers[0];
        arg->buf_out = buffers[1];
    }
    arg->program = program;
    arg->shader = shader;
}

void gl_setup_program(gl_args *arg, bool gen_buffers, LPCWSTR shader_file)
{
    gl_setup_program(arg, gen_buffers, shader_file, false);
}

void gl_read_buffer(cpx* dst, GLuint buffer, const int size)
{
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer);
    cpx* src = (cpx *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    memcpy(dst, src, sizeof(cpx) * size);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

void gl_setup(gl_args* a_dev, gl_args* a_host, cpx* in, cpx* out, int group_size, const int n)
{
    LPCWSTR shader_file_local = L"Platforms/OpenGL/gl_cshader_local.glsl";
    LPCWSTR shader_file_global = L"Platforms/OpenGL/gl_cshader_global.glsl";

    // "Local" compute shader
    a_dev->groups.x = (n >> 1) > group_size ? ((n >> 1) / group_size) : 1;
    a_dev->threads.x = (n >> 1) > group_size ? group_size : n >> 1;
    a_dev->number_of_blocks = a_dev->groups.x;
    gl_setup_program(a_dev, true, shader_file_local);

    // "Global" compute shader
    memcpy(a_host, a_dev, sizeof(gl_args));
    gl_setup_program(a_host, false, shader_file_global);

    // Load buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a_dev->buf_in);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(cpx), in, GL_STREAM_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, a_dev->buf_out);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(cpx), out, GL_STREAM_DRAW);
}

void gl_setup_2d(gl_args* a_dev, gl_args* a_host, gl_args* a_trans, cpx* in, cpx* out, int group_size, int tile_dim, const int n)
{
    LPCWSTR shader_file_local = L"Platforms/OpenGL/gl_cshader_local2d.glsl";
    LPCWSTR shader_file_global = L"Platforms/OpenGL/gl_cshader_global2d.glsl";
    LPCWSTR shader_file_trans = L"Platforms/OpenGL/gl_cshader_trans.glsl";

    // "Local" compute shader
    a_dev->groups.x = n;
    a_dev->groups.y = (n >> 1) > group_size ? (n >> 1) / group_size : 1;
    a_dev->threads.x = (n >> 1) > group_size ? group_size : n >> 1;
    a_dev->number_of_blocks = a_dev->groups.y;
    gl_setup_program(a_dev, true, shader_file_local);

    // "Global" compute shader
    memcpy(a_host, a_dev, sizeof(gl_args));
    gl_setup_program(a_host, false, shader_file_global);

    // "Transpose" compute shader
    memcpy(a_trans, a_dev, sizeof(gl_args));    
    gl_setup_program(a_trans, false, shader_file_trans, true);

    // Load buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a_dev->buf_in);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * n * sizeof(cpx), in, GL_STREAM_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, a_dev->buf_out);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * n * sizeof(cpx), out, GL_STREAM_DRAW);
}

void gl_shakedown(gl_args *a)
{
    if (a->shader_src != NULL)
        free(a->shader_src);
    if (a->buf_in && a->buf_out) {
        GLuint buffers[2];
        buffers[0] = a->buf_in;
        buffers[1] = a->buf_out;
        glDeleteBuffers(2, buffers);
        a->buf_in = 0;
        a->buf_out = 0;
    }
    glDeleteProgram(a->program);
    glDeleteShader(a->shader);
}