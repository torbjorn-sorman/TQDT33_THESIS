#ifndef MYTIMER_H
#define MYTIMER_H

#include <Windows.h>

static LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds, Frequency;

static void __inline startTimer()
{
    QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);
}

static double __inline stopTimer()
{
    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    return (double)ElapsedMicroseconds.QuadPart;
}

#endif