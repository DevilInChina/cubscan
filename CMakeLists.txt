cmake_minimum_required(VERSION 3.16)
project(cubscan CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cubscan main.cu warp_scab_shfl.cuh)

set_target_properties(cubscan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
