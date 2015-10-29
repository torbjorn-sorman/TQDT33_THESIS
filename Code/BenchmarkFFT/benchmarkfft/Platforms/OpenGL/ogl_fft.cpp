#include "ogl_fft.h"

#include <stdio.h>
#include <string>
/*
__inline void ogl_fft(transform_direction dir, ogl_args *args, const int n);
__inline void ogl_fft_2d(transform_direction dir, ogl_args *args, const int n);
*/
//
// Testing
//

void checkErrors(std::string desc) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL error in \"%s\": %s (%d)\n", desc.c_str(), gluErrorString(e), e);
        exit(20);
    }
}

GLuint genRenderProg(GLuint texHandle) {
    GLuint progHandle = glCreateProgram();
    GLuint vp = glCreateShader(GL_VERTEX_SHADER);
    GLuint fp = glCreateShader(GL_FRAGMENT_SHADER);

    const char *vpSrc[] = {
        "#version 430\n",
        "in vec2 pos;\
        		 out vec2 texCoord;\
                 		 void main() {\
                         			 texCoord = pos*0.5f + 0.5f;\
                                     			 gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);\
                                                 		 }"
    };

    const char *fpSrc[] = {
        "#version 430\n",
        "uniform sampler2D srcTex;\
        		 in vec2 texCoord;\
                 		 out vec4 color;\
                         		 void main() {\
                                 			 float c = texture(srcTex, texCoord).x;\
                                             			 color = vec4(c, 1.0, 1.0, 1.0);\
                                                         		 }"
    };

    glShaderSource(vp, 2, vpSrc, NULL);
    glShaderSource(fp, 2, fpSrc, NULL);

    glCompileShader(vp);
    int rvalue;
    glGetShaderiv(vp, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue) {
        fprintf(stderr, "Error in compiling vp\n");
        exit(30);
    }
    glAttachShader(progHandle, vp);

    glCompileShader(fp);
    glGetShaderiv(fp, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue) {
        fprintf(stderr, "Error in compiling fp\n");
        exit(31);
    }
    glAttachShader(progHandle, fp);

    glBindFragDataLocation(progHandle, 0, "color");
    glLinkProgram(progHandle);

    glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
    if (!rvalue) {
        fprintf(stderr, "Error in linking sp\n");
        exit(32);
    }

    glUseProgram(progHandle);
    glUniform1i(glGetUniformLocation(progHandle, "srcTex"), 0);

    GLuint vertArray;
    glGenVertexArrays(1, &vertArray);
    glBindVertexArray(vertArray);

    GLuint posBuf;
    glGenBuffers(1, &posBuf);
    glBindBuffer(GL_ARRAY_BUFFER, posBuf);
    float data[] = {
        -1.0f, -1.0f,
        -1.0f, 1.0f,
        1.0f, -1.0f,
        1.0f, 1.0f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8, data, GL_STREAM_DRAW);
    GLint posPtr = glGetAttribLocation(progHandle, "pos");
    glVertexAttribPointer(posPtr, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(posPtr);

    checkErrors("Render shaders");
    return progHandle;
}

GLuint genTexture() {
    // We create a single float channel 512^2 texture
    GLuint texHandle;
    glGenTextures(1, &texHandle);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texHandle);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 512, 512, 0, GL_RED, GL_FLOAT, NULL);

    // Because we're also using this tex as an image (in order to write to it),
    // we bind it to an image unit as well
    glBindImageTexture(0, texHandle, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    checkErrors("Gen texture");
    return texHandle;
}

GLuint genComputeProg(GLuint texHandle) {
    // Creating the compute shader, and the program object containing the shader
    GLuint progHandle = glCreateProgram();
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);

    // In order to write to a texture, we have to introduce it as image2D.
    // local_size_x/y/z layout variables define the work group size.
    // gl_GlobalInvocationID is a uvec3 variable giving the global ID of the thread,
    // gl_LocalInvocationID is the local index within the work group, and
    // gl_WorkGroupID is the work group's index
    const char *csSrc[] = {
        "#version 430\n",
        "uniform float roll;\
                 uniform image2D destTex;\
                          layout (local_size_x = 16, local_size_y = 16) in;\
                                   void main() {\
                                                ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);\
                                                             float localCoef = length(vec2(ivec2(gl_LocalInvocationID.xy)-8)/8.0);\
                                                                          float globalCoef = sin(float(gl_WorkGroupID.x+gl_WorkGroupID.y)*0.1 + roll)*0.5;\
                                                                                       imageStore(destTex, storePos, vec4(1.0-globalCoef*localCoef, 0.0, 0.0, 0.0));\
                                                                                                }"
    };

    glShaderSource(cs, 2, csSrc, NULL);
    glCompileShader(cs);
    int rvalue;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue) {
        fprintf(stderr, "Error in compiling the compute shader\n");
        GLchar log[10240];
        GLsizei length;
        glGetShaderInfoLog(cs, 10239, &length, log);
        fprintf(stderr, "Compiler log:\n%s\n", log);
        exit(40);
    }
    glAttachShader(progHandle, cs);

    glLinkProgram(progHandle);
    glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
    if (!rvalue) {
        fprintf(stderr, "Error in linking compute shader program\n");
        GLchar log[10240];
        GLsizei length;
        glGetProgramInfoLog(progHandle, 10239, &length, log);
        fprintf(stderr, "Linker log:\n%s\n", log);
        exit(41);
    }
    glUseProgram(progHandle);

    glUniform1i(glGetUniformLocation(progHandle, "destTex"), 0);

    checkErrors("Compute shader");
    return progHandle;
}

void testrun()
{
    GLuint renderHandle, computeHandle, texHandle;
    texHandle = genTexture();
    //renderHandle = genRenderProg(texHandle);
    computeHandle = genComputeProg(texHandle);
    //glutCreateWindow("SAXPY TESTS");
    //GLuint computeHandle = genComputeProg();
    
    //GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    /*
    const char *source_string[] = { "" };

    glShaderSource(cs, 2, source_string, NULL);
    glCompileShader(cs);
    int rvalue;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue) {
        printf("Error in compiling the compute shader\n");
        GLchar log[10240];
        GLsizei length;
        glGetShaderInfoLog(cs, 10239, &length, log);
        printf("Compiler log:\n%s\n", log);
        exit(40);
    }
    glAttachShader(progHandle, cs);

    glLinkProgram(progHandle);
    glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
    if (!rvalue) {
        fprintf(stderr, "Error in linking compute shader program\n");
        GLchar log[10240];
        GLsizei length;
        glGetProgramInfoLog(progHandle, 10239, &length, log);
        fprintf(stderr, "Linker log:\n%s\n", log);
        exit(41);
    }
    glUseProgram(progHandle);

    glUniform1i(glGetUniformLocation(progHandle, "destTex"), 0);
    */
}

int ogl_validate(const int n)
{
    testrun();
    /*
    cpx *in = get_seq(n, 1);
    cpx *out = get_seq(n);
    cpx *ref = get_seq(n, in);
    ogl_args args;
    ogl_setup(&args, in, n);

    ogl_fft(FFT_FORWARD, &args, n);
    ogl_read_buffer(&args, args.buf_output, out, n);
    double forward_diff = diff_forward_sinus(out, n);

    swap_io(&args);
    ogl_fft(FFT_INVERSE, &args, n);
    ogl_read_buffer(&args, args.buf_output, out, n);
    double inverse_diff = diff_seq(out, ref, n);

    ogl_shakedown(&args);
    free(in);
    free(out);
    free(ref);
    return forward_diff < RELATIVE_ERROR_MARGIN && inverse_diff < RELATIVE_ERROR_MARGIN;
    */
    return 0;
}

int ogl_2d_validate(const int n, bool write_img)
{
    /*
    cpx *data, *ref;
    setup_seq2D(&data, NULL, &ref, n);
    ogl_args args;
    ogl_setup_2d(&args, data, n);

    ogl_fft_2d(FFT_FORWARD, &args, n);
    if (write_img) {
    ogl_read_buffer(&args, args.buf_output, data, n * n);
    write_normalized_image("DirectX", "freq", data, n, true);
    }
    swap_io(&args);

    ogl_fft_2d(FFT_INVERSE, &args, n);
    ogl_read_buffer(&args, args.buf_output, data, n);
    if (write_img)
    write_image("DirectX", "spat", data, n);

    ogl_shakedown(&args);
    double diff = diff_seq(data, ref, n);
    free(data);
    free(ref);
    return diff < RELATIVE_ERROR_MARGIN;
    */
    return 0;
}

#ifndef MEASURE_BY_TIMESTAMP
double ogl_performance(const int n)
{
    double measures[NUM_TESTS];
    ogl_args args;
    for (int i = 0; i < NUM_TESTS; ++i) {
        ogl_setup(&args, NULL, n);
        startTimer();
        ogl_fft(FFT_FORWARD, &args, n);
        measures[i] = stopTimer();
        ogl_shakedown(&args);
    }
    return average_best(measures, NUM_TESTS);
}
double ogl_2d_performance(const int n)
{
    double measures[NUM_TESTS];
    ogl_args args;
    for (int i = 0; i < NUM_TESTS; ++i) {
        ogl_setup_2d(&args, NULL, n);
        startTimer();
        ogl_fft_2d(FFT_FORWARD, &args, n);
        measures[i] = stopTimer();
        ogl_shakedown(&args);
    }
    return average_best(measures, NUM_TESTS);
}
#else
double ogl_performance(const int n)
{
    /*
    ogl_args args;
    profiler_data profiler[NUM_TESTS];
    ogl_setup(&args, NULL, n);
    for (int i = 0; i < NUM_TESTS; ++i) {
    profiler_data p;
    ogl_start_profiling(&args, &p);

    ogl_fft(FFT_FORWARD, &args, n);

    ogl_end_profiling(&args, &p);
    profiler[i] = p;
    }
    ogl_shakedown(&args);
    return ogl_avg(profiler, &args);
    */
    return -1;
}
double ogl_2d_performance(const int n)
{
    /*
    ogl_args args;
    ogl_setup_2d(&args, NULL, n);
    profiler_data profiler[NUM_TESTS];
    for (int i = 0; i < NUM_TESTS; ++i) {
    profiler_data p;
    ogl_start_profiling(&args, &p);

    ogl_fft_2d(FFT_FORWARD, &args, n);

    ogl_end_profiling(&args, &p);
    profiler[i] = p;
    }
    ogl_shakedown(&args);
    return ogl_avg(profiler, &args);
    */
    return -1;
}
#endif

//
// Algorithm
//
/*
__inline void ogl_set_buffers(ogl_args *a)
{
a->context->CSSetShaderResources(0, 1, &a->view_nullptr);
a->context->CSSetUnorderedAccessViews(0, 1, &a->buf_output_uav, &a->init_cnts);
a->context->CSSetShaderResources(0, 1, &a->buf_input_srv);
}

__inline void ogl_set_local_args(ogl_args *a, float global_angle, float local_angle, int steps_left, int leading_bits, int steps_gpu, float scalar, int number_of_blocks, int block_range_half)
{
ogl_cs_args cb = { global_angle, local_angle, scalar, steps_left, leading_bits, steps_gpu, number_of_blocks, block_range_half, 0, 0, 0 };
ogl_map_args<ogl_cs_args>(a->context, a->buf_constant, &cb);
ogl_set_buffers(a);
a->context->CSSetShader(a->cs_local, nullptr, 0);
}

__inline void ogl_set_global_args(ogl_args *a, float angle, int dist, unsigned int lmask, int steps)
{
ogl_cs_args cb = { angle, 0.f, 0.f, 0, 0, 0, 0, 0, steps, lmask, dist };
ogl_map_args<ogl_cs_args>(a->context, a->buf_constant, &cb);
ogl_set_buffers(a);
a->context->CSSetShader(a->cs_global, nullptr, 0);
}

__inline void ogl_fft(transform_direction dir, ogl_args *args, const int n)
{
int n_half = (n >> 1);
int steps_left = log2_32(n);
int leading_bits = 32 - steps_left;
int steps_gpu = log2_32(MAX_BLOCK_SIZE);
float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
int number_of_blocks = n_half > MAX_BLOCK_SIZE ? (n_half / MAX_BLOCK_SIZE) : 1;;
int n_per_block = n / number_of_blocks;
float global_angle = dir * (M_2_PI / n);
float local_angle = dir * (M_2_PI / n_per_block);
int block_range_half = n_half;
if (number_of_blocks > 1) {
int steps = 0;
int dist = n;
while (--steps_left > steps_gpu) {
ogl_set_global_args(args, global_angle, dist >>= 1, 0xFFFFFFFF << steps_left, steps++);
args->context->Dispatch(args->n_groups.x, 1, 1);
swap_io(args);
}
++steps_left;
block_range_half = n_per_block >> 1;
}
ogl_set_local_args(args, global_angle, local_angle, steps_left, leading_bits, steps_gpu, scalar, 1, block_range_half);
args->context->Dispatch(args->n_groups.x, 1, 1);
}

__inline void ogl_fft_2d_helper(transform_direction dir, ogl_args *args, const int n)
{
int steps_left = log2_32(n);
int leading_bits = 32 - steps_left;
float scalar = (dir == FFT_FORWARD ? 1.f : 1.f / n);
const int n_per_block = n / args->n_groups.y;
const float global_angle = dir * (M_2_PI / n);
const float local_angle = dir * (M_2_PI / n_per_block);
int block_range = n;
if (args->n_groups.y > 1) {
const int steps_gpu = log2_32(MAX_BLOCK_SIZE);
ogl_cs_args cb = { global_angle, 0.f, 0.f, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF << steps_left, n };
args->context->CSSetShader(args->cs_global, nullptr, 0);
while (--steps_left > steps_gpu) {
cb.dist >>= 1;
cb.lmask = 0xFFFFFFFF << steps_left;
ogl_map_args<ogl_cs_args>(args->context, args->buf_constant, &cb);
ogl_set_buffers(args);
args->context->Dispatch(args->n_groups.x, args->n_groups.y, 1);
swap_io(args);
++cb.steps;
}
++steps_left;
block_range = n_per_block;
}
ogl_set_local_args(args, global_angle, local_angle, steps_left, leading_bits, 0, scalar, 0, block_range >> 1);
args->context->Dispatch(args->n_groups.x, args->n_groups.y, 1);
}

__inline void ogl_fft_2d(transform_direction dir, ogl_args *args, const int n)
{
UINT width = n > TILE_DIM ? (n / TILE_DIM) : 1;

ogl_fft_2d_helper(dir, args, n);

swap_io(args);
ogl_set_buffers(args);
args->context->CSSetShader(args->cs_transpose, nullptr, 0);
args->context->Dispatch(width, width, 1);

swap_io(args);
ogl_fft_2d_helper(dir, args, n);

swap_io(args);
ogl_set_buffers(args);
args->context->CSSetShader(args->cs_transpose, nullptr, 0);
args->context->Dispatch(width, width, 1);
}
*/