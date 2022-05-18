//
// Created by daudi on 16/05/22.
//

#ifndef MANDELBROT_MANDELBROT_H
#define MANDELBROT_MANDELBROT_H

#include <cmath>
#include <complex>
#include <iostream>
#include <utility>
#include <Magick++.h>

typedef struct GradientControlPoint {
    float stop;
    Magick::Color color;
    GradientControlPoint(float _stop, const Magick::Color& _color) : stop(_stop), color(_color) {}
} GradientControlPoint;

typedef struct RGBA {
    Magick::Quantum red;
    Magick::Quantum green;
    Magick::Quantum blue;
    Magick::Quantum alpha;
} rgba;

#endif //MANDELBROT_MANDELBROT_H
