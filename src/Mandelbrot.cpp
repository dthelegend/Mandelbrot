//
// Created by Daudi on 16/05/22.
//

#include "Mandelbrot.h"

Mandelbrot::Mandelbrot(Axes<int> cartesianBounds, Axes<double> argandBounds, int maxDepth)
    : maxDepth(maxDepth), Cartesian(cartesianBounds), Argand(argandBounds) {}

Point<double> Mandelbrot::CartesianToArgand(Point<int> cartesianPoint) const {
    return Point<double> {
        this->Argand.X.Start + ((double) (cartesianPoint.X - this->Cartesian.X.Start) / (this->Cartesian.X.End - this->Cartesian.X.Start)) * (this->Argand.X.End - this->Argand.X.Start),
        this->Argand.Y.Start + ((double) (cartesianPoint.Y - this->Cartesian.Y.Start) / (this->Cartesian.Y.End - this->Cartesian.Y.Start)) * (this->Argand.Y.End - this->Argand.Y.Start)
    };
}

double Mandelbrot::EvaluatePoint(Point<int> point) const {
    return (double) EvaluatePoint(CartesianToArgand(point)) / maxDepth;
}

int Mandelbrot::EvaluatePoint(Point<double> point) const {
    int depth;
    auto zn = Point<double>{0,0};
    for(depth = 0; depth < maxDepth && pow(zn.X,2) + pow(zn.Y,2) <= 4; ++depth) {
        zn.X = pow(zn.X,2) - pow(zn.Y, 2) + point.X;
        zn.Y = 2 * zn.X * zn.Y + point.Y;
    }
    return depth;
}

template <typename T>
bool Axes<T>::hasPoint(Point<T> point) const {
    return this->X.Start + point.X < this->X.End && this->Y.Start + point.X < this->Y.End;
}

struct BMPHeader {
    uint16_t header;
    uint32_t size;
    uint16_t reserved_0;
    uint16_t reserved_1;
};

int main(int argc, char *argv[]) {
    auto mandelbrot = new Mandelbrot(
        Axes<int>{
            Dimension<int>{0,1920},
            Dimension<int>{0,1080}
        },
        Axes<double>{
            Dimension<double>{0,1920},
            Dimension<double>{0,1080}
        },
        10
    );

    // Create file
    FILE * fp;
    if (!(fp = fopen("image.bmp", "wb"))) {
        return 3;
    }

    fwrite(, sizeof(char), 1, fp);
    fwrite(bmp_dib_v3_header, sizeof(bmp_dib_v3_header_t), 1, fp);

    Point<int> cursor{};
    for (cursor.X = 0; cursor.X < 1920; cursor.X++)  {
        for (cursor.Y = 0; cursor.Y < 1080; cursor.Y++) {
            int color = floor(255 * mandelbrot->EvaluatePoint(cursor));
            fwrite(&color, 1, 1, fp);
            fwrite(&color, 1, 1, fp);
            fwrite(&color, 1, 1, fp);
        }
    }

    /* cleanup heap allocation */
    fclose(fp);

    return 0;
}
