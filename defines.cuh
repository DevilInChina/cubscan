//
// Created by yxgao on 2022/2/28.
//

#ifndef CUBSCAN_DEFINES_CUH
#define CUBSCAN_DEFINES_CUH

#if ((__CUDACC_VER_MAJOR__ >= 9) || defined(_NVHPC_CUDA) || \
     CUDA_VERSION >= 9000) && \
  !defined(CUB_USE_COOPERATIVE_GROUPS)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

#ifndef CUB_LOG_WARP_THREADS
#define CUB_LOG_WARP_THREADS(arch)                      \
        (5)
#define CUB_WARP_THREADS(arch)                          \
        (1 << CUB_LOG_WARP_THREADS(arch))

#define CUB_PTX_WARP_THREADS        CUB_WARP_THREADS(CUB_PTX_ARCH)
#define CUB_PTX_LOG_WARP_THREADS    CUB_LOG_WARP_THREADS(CUB_PTX_ARCH)
#endif

#ifndef CUB_PTX_ARCH
#if defined(_NVHPC_CUDA)
// __NVCOMPILER_CUDA_ARCH__ is the target PTX version, and is defined
        // when compiling both host code and device code. Currently, only one
        // PTX version can be targeted.
#define CUB_PTX_ARCH __NVCOMPILER_CUDA_ARCH__
#elif !defined(__CUDA_ARCH__)
#define CUB_PTX_ARCH 0
#else
#define CUB_PTX_ARCH __CUDA_ARCH__
#endif
#endif

namespace scan {
    /**
 * \brief Returns the row-major linear thread identifier for a multidimensional thread block
 */
    __device__ __forceinline__ int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z) {
        return ((block_dim_z == 1) ? 0 : (threadIdx.z * block_dim_x * block_dim_y)) +
               ((block_dim_y == 1) ? 0 : (threadIdx.y * block_dim_x)) +
               threadIdx.x;
    }

}

#endif //CUBSCAN_DEFINES_CUH
