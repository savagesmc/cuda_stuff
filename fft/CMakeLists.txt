cmake_minimum_required(VERSION 3.8)
project(smc_fft_cuda LANGUAGES CXX CUDA)

include(CTest)

add_library(smc_fft_lib_cuda
   gpufft.cu
   gpufft.h
   )

target_compile_features(smc_fft_lib_cuda PUBLIC cxx_std_11)


add_executable(fft_test fft_tester.cpp)

set_property(TARGET fft_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(fft_test PRIVATE smc_fft_lib_cuda cufft)
