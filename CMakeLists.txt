cmake_minimum_required(VERSION 3.1)
project(UNIX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(stft-peaks stft-peaks.cpp)
add_executable(analysis-resynthesis analysis-resynthesis.cpp)
