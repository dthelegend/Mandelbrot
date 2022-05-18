//
// Created by Daudi on 16/05/22.
// export OverrideDefaultFP64Settings=1
// export IGC_EnableDPEmulation=1
// dpcpp `Magick++-config --cxxflags --cppflags` src/Mandelbrot.cpp `Magick++-config --ldflags --libs` -o a.o
//

#include "Mandelbrot.h"

static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
            std::cout << "Failure" << std::endl;
            std::terminate();
        }
    }
};

int main(int argc, char *argv[]) {
    // Parameters
    const float argandStartX = -2;
    const float argandEndX = 0.75;
    const float argandStartY = -0.7734375;
    const float argandEndY = 0.7734375;
    const unsigned int imageWidth = 15360;
    const unsigned int imageHeight = 8640;
    const unsigned long maxDepth = 1000;

    // Calculated
    const float argandRangeX = argandEndX - argandStartX;
    const float argandRangeY = argandEndY - argandStartY;

    std::cout << "Creating Sycl Job..." << std::endl;

    // Initialise device
    sycl::default_selector d_selector;

    // Create a command queue
    sycl::queue queue(d_selector, exception_handler);

    // Report on the device the queue is using
    std::cout << "Using device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    sycl::range<2> num_pixels{ imageWidth, imageHeight };

    sycl::buffer<unsigned int, 2> buffer(num_pixels);

    queue.submit([&](sycl::handler &h) {
        sycl::accessor buff_access(buffer, h, sycl::write_only);

        h.parallel_for(num_pixels, [=](sycl::id<2> i) {
            unsigned long x = i[0], y = i[1];
            unsigned int depth;

            std::complex<float> zn(0, 0);
            std::complex<float> c(((float) x / imageWidth) * argandRangeX + argandStartX, ((float) y / imageHeight) * argandRangeY + argandStartY);

            for (depth = 0; depth < maxDepth && abs(zn) <= 2; ++depth) {
                zn = pow(zn, 2) + c;
            }

            buff_access[x][y] = depth;
        });
    });

    std::cout << "Sycl kernels launched" << std::endl;

    queue.wait_and_throw();

    std::cout << "Sycl kernels completed" << std::endl;

    // Create Image
    std::cout << "Creating Image" << std::endl;

    Magick::InitializeMagick(*argv);

    sycl::host_accessor hostAccessor(buffer);

    auto image = new Magick::Image(Magick::Geometry(imageWidth, imageHeight), Magick::ColorRGB(0, 0, 0, 0));

    for(int i = 0; i < imageWidth; i++) {
        for(int j = 0; j < imageHeight; j++) {
        }
    }

    image->write("/home/daudi/Code/Mandelbrot/Mandelbrot.png");

    Magick::TerminateMagick();

    std::cout << "Image created" << std::endl;

    return 0;
}
