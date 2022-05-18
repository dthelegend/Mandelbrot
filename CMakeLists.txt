cmake_minimum_required(VERSION 3.23)
project(Mandelbrot)

#set(CMAKE_CXX_STANDARD 17)
find_package(IntelDPCPP REQUIRED)
find_package(ImageMagick REQUIRED COMPONENTS Magick++)
include_directories(src)
include_directories(${ImageMagick_INCLUDE_DIRS})

add_executable(Mandelbrot
        src/Mandelbrot.cpp
        src/Mandelbrot.h)

target_link_libraries(Mandelbrot OpenCL sycl ${ImageMagick_LIBRARIES})