
#define FREEGLUT_LIB_PRAGMAS 0

#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(500, 200);//optional
    glutInitWindowSize(800, 600); //optional
    glutCreateWindow("OpenGL First Window");

    getchar();

    glutMainLoop();
        
    return 0;
}