//
// Created by yxgao on 2022/2/28.
//
#include "cub_origin/cub.cuh"

#if (__CUDACC_VER_MAJOR__ >= 9 || CUDA_VERSION >= 9000) && !_NVHPC_CUDA

#include <cuda_fp16.h>

#endif
#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !_NVHPC_CUDA
#include <cuda_bf16.h>
#endif

#include <type_traits>
#include <limits>

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
#define SCAN_BLOCK_SIZE 128
#define SCAN_ITEM_PER_BLOCK 12
#define SCAN_TILE_SIZE (SCAN_BLOCK_SIZE*SCAN_ITEM_PER_BLOCK)
#define SCAN_TILE_STATUS_PADDING 32
#define SCAN_WARP_SIZE 32
    enum ScanTileStatus {
        SCAN_TILE_OOB,          // Out-of-bounds (e.g., padding)
        SCAN_TILE_INVALID = 99, // Not yet processed
        SCAN_TILE_PARTIAL,      // Tile aggregate is available
        SCAN_TILE_INCLUSIVE,    // Inclusive tile prefix is available
    };

    /**
     * \brief Returns the row-major linear thread identifier for a multidimensional thread block
     */
    __device__ __forceinline__ int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z) {
        return ((block_dim_z == 1) ? 0 : (threadIdx.z * block_dim_x * block_dim_y)) +
               ((block_dim_y == 1) ? 0 : (threadIdx.y * block_dim_x)) +
               threadIdx.x;
    }

    /**
     * \brief Allows for the treatment of an integral constant as a type at compile-time (e.g., to achieve static call dispatch based on constant integral values)
     */
    template<int A>
    struct Int2Type {
        enum {
            VALUE = A
        };
    };
    /**
     * \brief Statically determine log2(N), rounded up.
     *
     * For example:
     *     Log2<8>::VALUE   // 3
     *     Log2<3>::VALUE   // 2
     */
    template<int N, int CURRENT_VAL = N, int COUNT = 0>
    struct Log2 {
        /// Static logarithm value
        enum {
            VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE
        };         // Inductive case
    };


    template<int N, int COUNT>
    struct Log2<N, 0, COUNT> {
        enum {
            VALUE = (1 << (COUNT - 1) < N) ?                                  // Base case
                    COUNT :
                    COUNT - 1
        };
    };
    template<int N>
    struct PowerOfTwo {
        enum {
            VALUE = ((N & (N - 1)) == 0)
        };
    };
    namespace cub {
        struct Sum {
            /// Binary sum operator, returns <tt>a + b</tt>
            template<typename T>
            __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
                return a + b;
            }
        };
    }
    enum Category {
        NOT_A_NUMBER,
        SIGNED_INTEGER,
        UNSIGNED_INTEGER,
        FLOATING_POINT
    };
    template<Category _CATEGORY, bool _PRIMITIVE, bool _NULL_TYPE, typename _UnsignedBits, typename T>
    struct BaseTraits {
        /// Category
        static const Category CATEGORY = _CATEGORY;
        enum {
            PRIMITIVE = _PRIMITIVE,
            NULL_TYPE = _NULL_TYPE,
        };
    };
    template<typename T>
    struct AlignBytes {
        struct Pad {
            T val;
            char byte;
        };

        enum {
            /// The "true CUDA" alignment of T in bytes
            ALIGN_BYTES = sizeof(Pad) - sizeof(T)
        };

        /// The "truly aligned" type
        typedef T Type;
    };

    __device__ __forceinline__ unsigned int LaneMaskGe() {
        unsigned int ret;
        asm ("mov.u32 %0, %%lanemask_ge;" : "=r"(ret));
        return ret;
    }

    template<typename T>
    struct UnitWord {
        enum {
            ALIGN_BYTES = AlignBytes<T>::ALIGN_BYTES
        };

        template<typename Unit>
        struct IsMultiple {
            enum {
                UNIT_ALIGN_BYTES = AlignBytes<Unit>::ALIGN_BYTES,
                IS_MULTIPLE = (sizeof(T) % sizeof(Unit) == 0) && (int(ALIGN_BYTES) % int(UNIT_ALIGN_BYTES) == 0)
            };
        };

        /// Biggest shuffle word that T is a whole multiple of and is not larger than
        /// the alignment of T
        using ShuffleWord = std::conditional_t<
                IsMultiple<int>::IS_MULTIPLE,
                unsigned int,
                std::conditional_t<IsMultiple<short>::IS_MULTIPLE,
                        unsigned short,
                        unsigned char>>;

        /// Biggest volatile word that T is a whole multiple of and is not larger than
        /// the alignment of T
        using VolatileWord =
        std::conditional_t<IsMultiple<long long>::IS_MULTIPLE,
                unsigned long long,
                ShuffleWord>;

        /// Biggest memory-access word that T is a whole multiple of and is not larger
        /// than the alignment of T
        using DeviceWord =
        std::conditional_t<IsMultiple<longlong2>::IS_MULTIPLE,
                ulonglong2,
                VolatileWord>;

        /// Biggest texture reference word that T is a whole multiple of and is not
        /// larger than the alignment of T
        using TextureWord = std::conditional_t<
                IsMultiple<int4>::IS_MULTIPLE,
                uint4,
                std::conditional_t<IsMultiple<int2>::IS_MULTIPLE, uint2, ShuffleWord>>;
    };

    struct NullType {
        using value_type = NullType;

        template<typename T>
        __host__ __device__ __forceinline__ NullType &operator=(const T &) { return *this; }

        __host__ __device__ __forceinline__ bool operator==(const NullType &) { return true; }

        __host__ __device__ __forceinline__ bool operator!=(const NullType &) { return false; }
    };

    __device__ __forceinline__
    unsigned int SHFL_DOWN_SYNC(unsigned int word, int src_offset, int flags, unsigned int member_mask) {
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile("shfl.sync.down.b32 %0, %1, %2, %3, %4;"
        : "=r"(word) : "r"(word), "r"(src_offset), "r"(flags), "r"(member_mask));
#else
        asm volatile("shfl.down.b32 %0, %1, %2, %3;"
        : "=r"(word) : "r"(word), "r"(src_offset), "r"(flags));
#endif
        return word;
    }

    template<
            int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
            typename T>
    __device__ __forceinline__ T ShuffleDown(
            T input,              ///< [in] The value to broadcast
            int src_offset,         ///< [in] The relative up-offset of the peer to read from
            int last_thread,        ///< [in] Index of last thread in logical warp (typically 31 for a 32-thread warp)
            unsigned int member_mask)        ///< [in] 32-bit mask of participating warp lanes
    {
        /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
        enum {
            SHFL_C = (32 - LOGICAL_WARP_THREADS) << 8
        };

        typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

        const int WORDS = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

        T output;
        ShuffleWord *output_alias = reinterpret_cast<ShuffleWord *>(&output);
        ShuffleWord *input_alias = reinterpret_cast<ShuffleWord *>(&input);

        unsigned int shuffle_word;
        shuffle_word = SHFL_DOWN_SYNC((unsigned int) input_alias[0], src_offset, last_thread | SHFL_C, member_mask);
        output_alias[0] = shuffle_word;

#pragma unroll
        for (int WORD = 1; WORD < WORDS; ++WORD) {
            shuffle_word = SHFL_DOWN_SYNC((unsigned int) input_alias[WORD], src_offset, last_thread | SHFL_C,
                                          member_mask);
            output_alias[WORD] = shuffle_word;
        }

        return output;
    }

    /**
     * \brief Numeric type traits
     */
// clang-format off
    template<typename T>
    struct NumericTraits : BaseTraits<NOT_A_NUMBER, false, false, T, T> {
    };

    template<>
    struct NumericTraits<NullType> : BaseTraits<NOT_A_NUMBER, false, true, NullType, NullType> {
    };

    template<>
    struct NumericTraits<char> : BaseTraits<(std::numeric_limits<char>::is_signed) ? SIGNED_INTEGER
                                                                                   : UNSIGNED_INTEGER, true, false, unsigned char, char> {
    };
    template<>
    struct NumericTraits<signed char> : BaseTraits<SIGNED_INTEGER, true, false, unsigned char, signed char> {
    };
    template<>
    struct NumericTraits<short> : BaseTraits<SIGNED_INTEGER, true, false, unsigned short, short> {
    };
    template<>
    struct NumericTraits<int> : BaseTraits<SIGNED_INTEGER, true, false, unsigned int, int> {
    };
    template<>
    struct NumericTraits<long> : BaseTraits<SIGNED_INTEGER, true, false, unsigned long, long> {
    };
    template<>
    struct NumericTraits<long long> : BaseTraits<SIGNED_INTEGER, true, false, unsigned long long, long long> {
    };

    template<>
    struct NumericTraits<unsigned char> : BaseTraits<UNSIGNED_INTEGER, true, false, unsigned char, unsigned char> {
    };
    template<>
    struct NumericTraits<unsigned short> : BaseTraits<UNSIGNED_INTEGER, true, false, unsigned short, unsigned short> {
    };
    template<>
    struct NumericTraits<unsigned int> : BaseTraits<UNSIGNED_INTEGER, true, false, unsigned int, unsigned int> {
    };
    template<>
    struct NumericTraits<unsigned long> : BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long, unsigned long> {
    };
    template<>
    struct NumericTraits<unsigned long long>
            : BaseTraits<UNSIGNED_INTEGER, true, false, unsigned long long, unsigned long long> {
    };

    template<>
    struct NumericTraits<float> : BaseTraits<FLOATING_POINT, true, false, unsigned int, float> {
    };
    template<>
    struct NumericTraits<double> : BaseTraits<FLOATING_POINT, true, false, unsigned long long, double> {
    };
#if (__CUDACC_VER_MAJOR__ >= 9 || CUDA_VERSION >= 9000) && !_NVHPC_CUDA
    template<>
    struct NumericTraits<__half> : BaseTraits<FLOATING_POINT, true, false, unsigned short, __half> {
    };
#endif
#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !_NVHPC_CUDA
    template <> struct NumericTraits<__nv_bfloat16> :   BaseTraits<FLOATING_POINT, true, false, unsigned short, __nv_bfloat16> {};
#endif

    template<>
    struct NumericTraits<bool>
            : BaseTraits<UNSIGNED_INTEGER, true, false, typename UnitWord<bool>::VolatileWord, bool> {
    };
// clang-format on

/**
 * \brief Type traits
 */
    template<typename T>
    struct Traits : NumericTraits<typename std::remove_cv<T>::type> {
    };
}

#endif //CUBSCAN_DEFINES_CUH
