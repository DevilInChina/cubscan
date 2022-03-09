cmake_minimum_required(VERSION 3.16)
project(cubscan CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cubscan main.cu warp_scan_shfl.cuh defines.cuh block_scan.cuh do_scan.cuh warp_reduce_shfl.cuh scan_status.cuh)

set_target_properties(cubscan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
