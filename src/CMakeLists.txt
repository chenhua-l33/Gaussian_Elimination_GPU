cmake_minimum_required(VERSION 3.7)
project(GPU_Gaussian CXX)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

find_package(OpenCL REQUIRED)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin )


add_subdirectory(gaussianElimination)
add_subdirectory(optimizedGaussianElimination)
add_subdirectory(GaussianEliminationCI)
