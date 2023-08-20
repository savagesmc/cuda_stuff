cmake_minimum_required(VERSION 3.8)
project(smc_fft_cuda LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 14)

include(CTest)

add_library(smc_fft_lib_cuda
   GpuFft.cu
   GpuFft.h
   Fft.cpp
   Fft.h
   )

set_target_properties(smc_fft_lib_cuda PROPERTIES CUDA_ARCHITECTURES "61;75")
# target_compile_features(smc_fft_lib_cuda PUBLIC cxx_std_11)

add_executable(fft_test fft_tester.cpp)

set_target_properties(fft_test PROPERTIES CUDA_ARCHITECTURES "61;75")
set_property(TARGET fft_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(fft_test PRIVATE smc_fft_lib_cuda cufft)
