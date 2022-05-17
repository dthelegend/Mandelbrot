cmake_minimum_required(VERSION 3.23)
project(Mandlebrot)

set(CMAKE_CXX_STANDARD 17)
find_package(IntelDPCPP REQUIRED)
include_directories(src)

add_executable(Mandlebrot
        src/Mandelbrot.cpp
        src/Mandelbrot.h)
