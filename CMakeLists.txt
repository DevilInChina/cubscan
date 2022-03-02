cmake_minimum_required(VERSION 3.16)
project(cubscan CUDA)

set(CMAKE_CUDA_STANDARD 14)



add_executable(cubscan main.cu warp_scan_shfl.cuh config/defines.cuh block_scan.cuh single_pass_scan_operators.cuh warp_reduce.cuh warp_reduce_shfl.cuh)

set_target_properties(cubscan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
