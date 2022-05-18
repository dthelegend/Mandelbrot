//
// Created by Daudi on 16/05/22.
// export OverrideDefaultFP64Settings=1
// export IGC_EnableDPEmulation=1
// dpcpp `Magick++-config --cxxflags --cppflags` src/Mandelbrot.cpp `Magick++-config --ldflags --libs` -o a.o
//

#include "Mandelbrot.h"
#include "dpc_common.hpp"

int main(int argc, char *argv[]) {
    // Parameters
    const float argandStartX = -2;
    const float argandEndX = 0.75;
    const float argandStartY = -0.7734375;
    const float argandEndY = 0.7734375;
    const unsigned int imageWidth = 15360;
    const unsigned int imageHeight = 8640;
    const unsigned long maxDepth = 1000;
    const Magick::Color hue = Magick::ColorRGB(0,0.5,0);

    // Calculated
    const float argandRangeX = argandEndX - argandStartX;
    const float argandRangeY = argandEndY - argandStartY;
    const Magick::Quantum hueRed = hue.quantumRed();
    const Magick::Quantum hueBlue = hue.quantumBlue();
    const Magick::Quantum hueGreen = hue.quantumGreen();
    const Magick::Quantum hueAlpha = hue.quantumAlpha();

    // Create Image
    std::cout << "Creating Image" << std::endl;

    Magick::InitializeMagick(*argv);

    Magick::Image image(Magick::Geometry(imageWidth, imageHeight), Magick::ColorRGB(0, 0, 0, 0));

    image.type(Magick::TrueColorAlphaType);
    image.modifyImage();

    Magick::Pixels view(image);

    // Get pixels for buffer
    Magick::Quantum *pixels = view.set(0, 0, imageWidth, imageHeight);

    {
        std::cout << "Creating Sycl Job..." << std::endl;

        // Initialise device
        sycl::default_selector d_selector;

        // Create a command queue
        sycl::queue queue(d_selector);

        // Report on the device the queue is using
        std::cout << "Using device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

        sycl::buffer<Magick::Quantum, 3> buffer(pixels, {imageHeight, imageWidth, 4});

        queue.submit([&](sycl::handler &h) {
            sycl::accessor buff_access(buffer, h, sycl::write_only, sycl::no_init);

            h.parallel_for({view.columns(), view.rows()}, [=](sycl::id<2> i) {
                unsigned long x = i[0], y = i[1];
                unsigned int depth;

                std::complex<float> zn(0, 0);
                std::complex<float> c(((float) x / imageWidth) * argandRangeX + argandStartX,
                                      ((float) y / imageHeight) * argandRangeY + argandStartY);

                for (depth = 0; depth < maxDepth && abs(zn) <= 2; ++depth) {
                    zn = pow(zn, 2) + c;
                }

                if (depth < maxDepth){
                    auto percentageHue = (float) depth / (maxDepth - 1);
                    buff_access[y][x][0] = hueRed + (QuantumRange - hueRed) * percentageHue;
                    buff_access[y][x][1] = hueGreen + (QuantumRange - hueGreen) * percentageHue;
                    buff_access[y][x][2] = hueBlue + (QuantumRange - hueBlue) * percentageHue;
                }
                buff_access[y][x][3] = hueAlpha;
            });
        });

        std::cout << "Sycl kernels launched" << std::endl;

        queue.wait_and_throw();

        std::cout << "Sycl kernels completed" << std::endl;
    }

    std::cout << "Writing image from cache" << std::endl;

    view.sync();

    std::cout << "Writing image to disk" << std::endl;

    image.write("/home/daudi/Code/Mandelbrot/Mandelbrot.png");

    Magick::TerminateMagick();

    std::cout << "Image written to disk" << std::endl;

    return 0;
}
