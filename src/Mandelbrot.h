//
// Created by daudi on 16/05/22.
//

#ifndef MANDELBROT_MANDELBROT_H
#define MANDELBROT_MANDELBROT_H

#include <cmath>
#include <iostream>
#include <CL/sycl.hpp>
#include <libpng16/png.h>

template <typename T>
struct Dimension {
public:
    T Start;
    T End;
} ;

template <typename T>
struct Point {
public:
    T X;
    T Y;
} ;

template <typename T>
struct Axes {
public:
    Dimension<T> X;
    Dimension<T> Y;
    [[nodiscard]] bool hasPoint(Point<T> point) const;
} ;

class Mandelbrot {
    Axes<int> Cartesian;
    Axes<double> Argand;
    int maxDepth;
private:
    [[nodiscard]] Point<double> CartesianToArgand(Point<int> cartesianPoint) const;
    [[nodiscard]] int EvaluatePoint(Point<double> point) const;
public:
    Mandelbrot(Axes<int> cartesianBounds, Axes<double> argandBounds, int maxDepth);
    [[nodiscard]] double EvaluatePoint(Point<int> point) const;
};

#endif //MANDELBROT_MANDELBROT_H
