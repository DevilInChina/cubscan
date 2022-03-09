//
// Created by yxgao on 2022/2/26.
//
/**
 * @brief 使用warp完成一个block的前缀和计算。
 */
#include "defines.cuh"
#ifndef CUBSCAN_WARN_SCAB_SHFL_CUH
#define CUBSCAN_WARN_SCAB_SHFL_CUH
namespace scan {



    template<int LOGICAL_WARP_THREADS,
            int PTX_ARCH = CUB_PTX_ARCH>
    __host__ __device__ __forceinline__
    unsigned int WarpMask(unsigned int warp_id) {
        constexpr bool is_pow_of_two = PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE;
        constexpr bool is_arch_warp = LOGICAL_WARP_THREADS ==
                                      CUB_WARP_THREADS(PTX_ARCH);

        unsigned int member_mask =
                0xFFFFFFFFu >> (CUB_WARP_THREADS(PTX_ARCH) - LOGICAL_WARP_THREADS);

        if (is_pow_of_two && !is_arch_warp) {
            member_mask <<= warp_id * LOGICAL_WARP_THREADS;
        }

        return member_mask;
    }

    __device__ __forceinline__ unsigned int LaneId() {
        unsigned int ret;
        asm ("mov.u32 %0, %%laneid;" : "=r"(ret));
        return ret;
    }

    template<typename T, int LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS, int PTX_ARCH = CUB_PTX_ARCH>
    struct warp_scan_shfl {

        enum {
            /// Whether the logical warp size and the PTX warp size coincide
            IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(PTX_ARCH)),

            /// The number of warp scan steps
            STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

            /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
            SHFL_C = (CUB_WARP_THREADS(PTX_ARCH) - LOGICAL_WARP_THREADS) << 8
        };

        /// Lane index in logical warp
        unsigned int lane_id;

        /// Logical warp index in 32-thread physical warp
        unsigned int warp_id;

        /// 32-thread physical warp member mask of logical warp
        unsigned int member_mask;

        explicit __device__ __forceinline__
        warp_scan_shfl()
                : lane_id(LaneId()), warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS)),
                  member_mask(WarpMask<LOGICAL_WARP_THREADS, PTX_ARCH>(warp_id)) {
            if (!IS_ARCH_WARP) {
                lane_id = lane_id % LOGICAL_WARP_THREADS;
            }
        }

        /// Inclusive prefix scan step (specialized for summation across int32 types)
        __device__ __forceinline__ int InclusiveScanStep(
                int input,              ///< [in] Calling thread's input item.

                int first_lane,         ///< [in] Index of first lane in segment
                int offset)             ///< [in] Up-offset to pull from
        {
            int output;
            int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

            // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
            asm volatile(
                "{"
                "  .reg .s32 r0;"
                "  .reg .pred p;"
                "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
                "  @p add.s32 r0, r0, %4;"
                "  mov.s32 %0, r0;"
                "}"
                : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));
#else
            asm volatile(
            "{"
            "  .reg .s32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.s32 r0, r0, %4;"
            "  mov.s32 %0, r0;"
            "}"
            : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input));
#endif

            return output;
        }

        /// Inclusive prefix scan step (specialized for summation across uint32 types)
        __device__ __forceinline__ unsigned int InclusiveScanStep(
                unsigned int input,              ///< [in] Calling thread's input item.

                int first_lane,         ///< [in] Index of first lane in segment
                int offset)             ///< [in] Up-offset to pull from
        {
            unsigned int output;
            int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

            // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
            asm volatile(
                "{"
                "  .reg .u32 r0;"
                "  .reg .pred p;"
                "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
                "  @p add.u32 r0, r0, %4;"
                "  mov.u32 %0, r0;"
                "}"
                : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));
#else
            asm volatile(
            "{"
            "  .reg .u32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.u32 r0, r0, %4;"
            "  mov.u32 %0, r0;"
            "}"
            : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input));
#endif

            return output;
        }


        /// Inclusive prefix scan step (specialized for summation across fp32 types)
        __device__ __forceinline__ float InclusiveScanStep(
                float input,              ///< [in] Calling thread's input item.

                int first_lane,         ///< [in] Index of first lane in segment
                int offset)             ///< [in] Up-offset to pull from
        {
            float output;
            int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

            // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
            asm volatile(
                "{"
                "  .reg .f32 r0;"
                "  .reg .pred p;"
                "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
                "  @p add.f32 r0, r0, %4;"
                "  mov.f32 %0, r0;"
                "}"
                : "=f"(output) : "f"(input), "r"(offset), "r"(shfl_c), "f"(input), "r"(member_mask));
#else
            asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(shfl_c), "f"(input));
#endif

            return output;
        }


        /// Inclusive prefix scan step (specialized for summation across unsigned long long types)
        __device__ __forceinline__ unsigned long long InclusiveScanStep(
                unsigned long long input,              ///< [in] Calling thread's input item.
                int first_lane,         ///< [in] Index of first lane in segment
                int offset)             ///< [in] Up-offset to pull from
        {
            unsigned long long output;
            int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

            // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
            asm volatile(
                "{"
                "  .reg .u64 r0;"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
                "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
                "  mov.b64 r0, {lo, hi};"
                "  @p add.u64 r0, r0, %4;"
                "  mov.u64 %0, r0;"
                "}"
                : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));
#else
            asm volatile(
            "{"
            "  .reg .u64 r0;"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.u64 r0, r0, %4;"
            "  mov.u64 %0, r0;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input));
#endif

            return output;
        }


        /// Inclusive prefix scan step (specialized for summation across long long types)
        __device__ __forceinline__ long long InclusiveScanStep(
                long long input,              ///< [in] Calling thread's input item.

                int first_lane,         ///< [in] Index of first lane in segment
                int offset)             ///< [in] Up-offset to pull from
        {
            long long output;
            int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

            // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
            asm volatile(
                "{"
                "  .reg .s64 r0;"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
                "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
                "  mov.b64 r0, {lo, hi};"
                "  @p add.s64 r0, r0, %4;"
                "  mov.s64 %0, r0;"
                "}"
                : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));
#else
            asm volatile(
            "{"
            "  .reg .s64 r0;"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.s64 r0, r0, %4;"
            "  mov.s64 %0, r0;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input));
#endif

            return output;
        }


        /// Inclusive prefix scan step (specialized for summation across fp64 types)
        __device__ __forceinline__ double InclusiveScanStep(
                double input,              ///< [in] Calling thread's input item.

                int first_lane,         ///< [in] Index of first lane in segment
                int offset)             ///< [in] Up-offset to pull from
        {
            double output;
            int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

            // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
            asm volatile(
                "{"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  .reg .f64 r0;"
                "  mov.b64 %0, %1;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.sync.up.b32 lo|p, lo, %2, %3, %4;"
                "  shfl.sync.up.b32 hi|p, hi, %2, %3, %4;"
                "  mov.b64 r0, {lo, hi};"
                "  @p add.f64 %0, %0, r0;"
                "}"
                : "=d"(output) : "d"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));
#else
            asm volatile(
            "{"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  .reg .f64 r0;"
            "  mov.b64 %0, %1;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.f64 %0, %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(shfl_c));
#endif

            return output;
        }

        __device__ __forceinline__ void InclusiveScan(
                T input,              ///< [in] Calling thread's input item.
                T &inclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
                T &exclusive_output
        )            ///< [in] Binary scan operator
        {
            inclusive_output = input;

            // Iterate scan steps
            int segment_first_lane = 0;

            // Iterate scan steps
#pragma unroll
            for (int STEP = 0; STEP < 5; STEP++) {
                inclusive_output = InclusiveScanStep(
                        inclusive_output,
                        segment_first_lane,
                        (1 << STEP));
            }
            exclusive_output = inclusive_output - input;
        }
    };
}


#endif //CUBSCAN_WARN_SCAB_SHFL_CUH
