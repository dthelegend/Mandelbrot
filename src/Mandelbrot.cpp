//
// Created by Daudi on 16/05/22.
//

#include <queue>
#include "Mandelbrot.h"
#include "dpc_common.hpp"

int main(int argc, char *argv[]) {
    // Parameters
//    const float argandStartX = -2;
//    const float argandEndX = 0.75;
//    const float argandStartY = -0.7734375;
//    const float argandEndY = 0.7734375;
    const float argandStartX = -2.5;
    const float argandEndX = 1;
    const float argandStartY = -1.46875;
    const float argandEndY = 0.5;
    const unsigned int imageWidth = 15360;
    const unsigned int imageHeight = 8640;
    const unsigned long maxDepth = 1000;

    // Colors for the gradient
    std::array<GradientControlPoint, 6> gradients {
            GradientControlPoint{0,      Magick::ColorRGB(0, 0.027, .392)},
            GradientControlPoint{0.16,   Magick::ColorRGB(.125, .420, .796)},
            GradientControlPoint{0.42,   Magick::ColorRGB(.93, 1, 1)},
            GradientControlPoint{0.6425, Magick::ColorRGB(1, .667, 0)},
            GradientControlPoint{0.8575, Magick::ColorRGB(0, 0.008, 0)},
            GradientControlPoint{1,      Magick::ColorRGB(0, 0.027, .392)}
    };

    std::sort(std::begin(gradients), std::begin(gradients), [&](const GradientControlPoint& o0, const GradientControlPoint& o1) { return o0.stop > o1.stop; });


    // Calculated
    const float argandRangeX = argandEndX - argandStartX;
    const float argandRangeY = argandEndY - argandStartY;
    std::array<RGBA, 2048> gradientControlPoints{};

    for(int k = 0, currentStop = 0; k < gradientControlPoints.size(); ++k) {
        auto k_ratio = (float) k / gradientControlPoints.size();
        auto l = (k_ratio - gradients[currentStop].stop) / (gradients[currentStop + 1].stop - gradients[currentStop].stop);

        gradientControlPoints[k].red = gradients[currentStop].color.quantumRed() + (gradients[currentStop + 1].color.quantumRed() - gradients[currentStop].color.quantumRed()) * l;
        gradientControlPoints[k].green = gradients[currentStop].color.quantumGreen() + (gradients[currentStop + 1].color.quantumGreen() - gradients[currentStop].color.quantumGreen()) * l;
        gradientControlPoints[k].blue = gradients[currentStop].color.quantumBlue() + (gradients[currentStop + 1].color.quantumBlue() - gradients[currentStop].color.quantumBlue()) * l;
        gradientControlPoints[k].alpha = gradients[currentStop].color.quantumAlpha() + (gradients[currentStop + 1].color.quantumAlpha() - gradients[currentStop].color.quantumAlpha()) * l;

        if(gradients[currentStop + 1].stop < k_ratio) {
            ++currentStop;
        }
    }

    // Create Image
    std::cout << "Creating Image" << std::endl;

    Magick::InitializeMagick(*argv);

    Magick::Image image(Magick::Geometry(imageWidth, imageHeight), Magick::ColorRGB(0, 0, 0, 1));

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
        sycl::buffer<RGBA, 1> gradientControlPoints_buffer(gradientControlPoints);

        queue.submit([&](sycl::handler &h) {
            sycl::accessor buff_access(buffer, h, sycl::write_only, sycl::no_init);
            sycl::accessor GradientControlPoint_buff_access(gradientControlPoints_buffer, h, sycl::read_only);

            h.parallel_for({view.columns(), view.rows()}, [=](sycl::id<2> i) {
                unsigned long x = i[0], y = i[1];
                unsigned int depth;

                std::complex<float> zn(0, 0);
                std::complex<float> c(((float) x / imageWidth) * argandRangeX + argandStartX,
                                      ((float) y / imageHeight) * argandRangeY + argandStartY);

                for (depth = 0; depth < maxDepth && abs(zn) <= 2; ++depth) {
//                    zn = pow(zn, 2) + c; // Mandelbrot
                    zn = pow(std::complex<float>{abs(zn.real()), abs(zn.imag())}, 2) + c; // Burning Ship
                }

                if(depth < maxDepth) {
                    const auto& color = GradientControlPoint_buff_access[(int) (((float) depth / maxDepth) * (float) GradientControlPoint_buff_access.size())];

                    buff_access[y][x][0] = color.red;
                    buff_access[y][x][1] = color.green;
                    buff_access[y][x][2] = color.blue;
                    buff_access[y][x][3] = color.alpha;
                }
            });
        });

        std::cout << "Sycl kernels launched" << std::endl;

        queue.wait_and_throw();

        std::cout << "Sycl kernels completed" << std::endl;
    }

    std::cout << "Writing image from cache" << std::endl;

    view.sync();

    std::cout << "Writing image to disk" << std::endl;

    image.write("/home/daudi/Code/Mandelbrot/Burning Ship.png");

    Magick::TerminateMagick();

    std::cout << "Image written to disk" << std::endl;

    return 0;
}
