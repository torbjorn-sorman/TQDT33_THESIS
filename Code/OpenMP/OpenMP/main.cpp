/******************************************************************************
* FILE: main.c
* DESCRIPTION:
*   OpenMP Example - Hello World - C/C++ Version
*   In this simple example, the master thread forks a parallel region.
*   All threads in the team obtain their unique thread number and print it.
*   The master thread only prints the total number of threads.  Two OpenMP
*   library routines are used to obtain the number of threads and each
*   thread's number.
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>

#define _OPENMP_NOFORCE_MANIFEST

#define NUM_ELEMENTS 1024 * 64

int main(int argc, char *argv[])
{
    LARGE_INTEGER freq;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double tMCPU, tSCPU;

    // get ticks per second
    QueryPerformanceFrequency(&freq);

    int nthreads, tid, offset;

    int a[NUM_ELEMENTS], b[NUM_ELEMENTS], c[NUM_ELEMENTS];
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        a[i] = i;
        b[i] = NUM_ELEMENTS - i;
    }
    bool success = true;

    // Multicore
    QueryPerformanceCounter(&t1);
    /* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel private(nthreads, tid, offset)
    {
        /* Obtain thread number */
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
        for (int i = 0; i < NUM_ELEMENTS; i = i + nthreads) {
            offset = i + tid;
            c[offset] = (a[offset] + b[offset]);
        }
    }  /* All threads join master thread and disband */
    QueryPerformanceCounter(&t2);
    tMCPU = (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;

    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        if (NUM_ELEMENTS != c[i])
        {
            printf("MCPU fail at %d", i);
            success = false;
        }
    }

    // Singlecore
    QueryPerformanceCounter(&t1);
    for (int i = 0; i < NUM_ELEMENTS; ++i)
        c[i] = (a[i] + b[i]);
    QueryPerformanceCounter(&t2);
    tSCPU = (t2.QuadPart - t1.QuadPart) * 1000.0 / freq.QuadPart;

    
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        if (NUM_ELEMENTS != c[i])
        {
            printf("SCPU fail at %d", i);
            success = false;
        }
    }
    
    if (success) 
    {
        printf("Success\n");
        printf("Time M: %f, S: %f\n", tMCPU, tSCPU);
    } 
    else
        printf("Fail\n");

#ifdef _DEBUG
    getchar();
#endif
    return 0;
}