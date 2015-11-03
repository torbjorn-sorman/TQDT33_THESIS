#include "gl_helper.h"

void gl_swap_io(gl_args *a)
{
    GLuint buf = a->buf_out;
    a->buf_out = a->buf_in;
    a->buf_in = buf;
}

void gl_setup_program(gl_args *a, std::string shader_file)
{
    GLuint program = glCreateProgram();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    GLint length;
    a->shader_src = get_kernel_src(shader_file.c_str(), &length);
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
        exit(41);
    }
    GLuint buffers[2];
    glGenBuffers(a->number_of_buffers, buffers);
    a->buf_in = buffers[0];
    a->buf_out = buffers[1];
    a->program = program;
}

void gl_load_input(GLuint buffer, cpx* data, const int n)
{    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(cpx), data, GL_STREAM_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, buffer);
}

void gl_read_buffer(GLuint buffer, cpx* data, const int n)
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    data = (cpx *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
}

void gl_setup(gl_args* args, cpx* data, const int n)
{
    gl_setup_program(args, "Platforms/OpenGL/gl_cs.glsl");
    if (data != NULL)
        gl_load_input(args->buf_in, data, n);
}

void gl_shakedown(gl_args *a)
{
    if (a->shader_src != NULL)
        free(a->shader_src);
}

void testrun()
{
    GLuint program = glCreateProgram();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

    //std::string str_source = getKernel(filename.c_str());
    std::string str_source = { "#version 430\n\
                               #define width 16\n\
                               #define height 16\n\
                               layout(std430, binding = 5) buffer bbs{ int bs[]; };\n\
                               layout(local_size_x = width, local_size_y = height) in;\n\
                               void main()\n\
                               {\n\
                                   int i = int(gl_LocalInvocationID.x * 2);\n\
                                       bs[gl_LocalInvocationID.x] = -bs[gl_LocalInvocationID.x];\n\
                                       }" };
    const GLint length = str_source.size() + 1;

    char *src = (char *)malloc(sizeof(char) * (str_source.size() + 1));
    strcpy_s(src, sizeof(char) * (str_source.size() + 1), str_source.c_str());
    src[str_source.size()] = '\0';


    glShaderSource(shader, 1, &src, &length);
    glCompileShader(shader);

    int rvalue;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue) {
        fprintf(stderr, "Error in compiling the compute shader\n");
        GLchar log[10240];
        GLsizei length;
        glGetShaderInfoLog(shader, 10239, &length, log);
        fprintf(stderr, "Compiler log:\n%s\n", log);
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
        exit(41);
    }
    glUseProgram(program);

    GLuint ssbo; //Shader Storage Buffer Object
    int buf[16] = { 1, 2, -3, 4, 5, -6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    int *ptr;

    // Create buffer, upload data
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16 * sizeof(int), &buf, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssbo);

    //glUseProgram(program);
    glDispatchCompute(16, 1, 1);

    // Get data back!
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,
        ssbo);
    ptr = (int *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    printf("\nRESULT:\n");
    for (int i = 0; i < 16; i++)
    {
        printf("%d\n", ptr[i]);
    }
    getchar();
}