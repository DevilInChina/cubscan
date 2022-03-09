cmake_minimum_required(VERSION 3.16)
project(cubscan CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cubscan main.cu warp_scan_shfl.cuh defines.cuh block_scan.cuh do_scan.cuh warp_reduce_shfl.cuh scan_status.cuh)
add_executable(origin_cub_scan test_cub.cu )

set_target_properties(cubscan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(origin_cub_scan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

