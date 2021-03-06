//
// Created by yxgao on 2022/3/7.
//


#include <stdint.h>
#include "defines.cuh"
#ifndef CUBSCAN_WARP_REDUCE_SHFL_CUH
#define CUBSCAN_WARP_REDUCE_SHFL_CUH


namespace scan{


/**
 * \brief WarpReduceShfl provides SHFL-based variants of parallel reduction of items partitioned across a CUDA thread warp.
 *
 * LOGICAL_WARP_THREADS must be a power-of-two
 */
template <
        typename    T,                      ///< Data type being reduced
        int         LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
        int         PTX_ARCH>               ///< The PTX compute capability for which to to specialize this collective
struct WarpReduceShfl
{
    static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE,
    "LOGICAL_WARP_THREADS must be a power of two");

    //---------------------------------------------------------------------
    // Constants and type definitions
    //---------------------------------------------------------------------

    enum
    {
        /// Whether the logical warp size and the PTX warp size coincide
        IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(PTX_ARCH)),

        /// The number of warp reduction steps
        STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

        /// Number of logical warps in a PTX warp
        LOGICAL_WARPS = CUB_WARP_THREADS(PTX_ARCH) / LOGICAL_WARP_THREADS,

        /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
        SHFL_C = (CUB_WARP_THREADS(PTX_ARCH) - LOGICAL_WARP_THREADS) << 8

    };

    template <typename S>
    struct IsInteger
    {
        enum {
            ///Whether the data type is a small (32b or less) integer for which we can use a single SFHL instruction per exchange
            IS_SMALL_UNSIGNED = (Traits<S>::CATEGORY == UNSIGNED_INTEGER) && (sizeof(S) <= sizeof(unsigned int))
        };
    };



    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    /// Lane index in logical warp
    int lane_id;

    /// Logical warp index in 32-thread physical warp
    int warp_id;

    /// 32-thread physical warp member mask of logical warp
    uint32_t member_mask;


    //---------------------------------------------------------------------
    // Construction
    //---------------------------------------------------------------------

    /// Constructor
    __device__ __forceinline__ WarpReduceShfl()
            : lane_id(static_cast<int>(LaneId()))
            , warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS))
            , member_mask(WarpMask<LOGICAL_WARP_THREADS, PTX_ARCH>(warp_id))
    {
        if (!IS_ARCH_WARP)
        {
            lane_id = lane_id % LOGICAL_WARP_THREADS;
        }
    }


    //---------------------------------------------------------------------
    // Reduction steps
    //---------------------------------------------------------------------

    /// Reduction (specialized for summation across uint32 types)
    __device__ __forceinline__ unsigned int ReduceStep(
            unsigned int    input,              ///< [in] Calling thread's input item.
            cub::Sum        /*reduction_op*/,   ///< [in] Binary reduction operator
            int             last_lane,          ///< [in] Index of last lane in segment
            int             offset)             ///< [in] Up-offset to pull from
    {
        unsigned int output;
        int shfl_c = last_lane | SHFL_C;   // Shuffle control (mask and last_lane)

        // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
            "{"
            "  .reg .u32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.down.b32 r0|p, %1, %2, %3, %5;"
            "  @p add.u32 r0, r0, %4;"
            "  mov.u32 %0, r0;"
            "}"
            : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));
#else
        asm volatile(
        "{"
        "  .reg .u32 r0;"
        "  .reg .pred p;"
        "  shfl.down.b32 r0|p, %1, %2, %3;"
        "  @p add.u32 r0, r0, %4;"
        "  mov.u32 %0, r0;"
        "}"
        : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input));
#endif

        return output;
    }


    /// Reduction (specialized for summation across fp32 types)
    __device__ __forceinline__ float ReduceStep(
            float           input,              ///< [in] Calling thread's input item.
            cub::Sum        /*reduction_op*/,   ///< [in] Binary reduction operator
            int             last_lane,          ///< [in] Index of last lane in segment
            int             offset)             ///< [in] Up-offset to pull from
    {
        float output;
        int shfl_c = last_lane | SHFL_C;   // Shuffle control (mask and last_lane)

        // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.down.b32 r0|p, %1, %2, %3, %5;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(shfl_c), "f"(input), "r"(member_mask));
#else
        asm volatile(
        "{"
        "  .reg .f32 r0;"
        "  .reg .pred p;"
        "  shfl.down.b32 r0|p, %1, %2, %3;"
        "  @p add.f32 r0, r0, %4;"
        "  mov.f32 %0, r0;"
        "}"
        : "=f"(output) : "f"(input), "r"(offset), "r"(shfl_c), "f"(input));
#endif

        return output;
    }


    /// Reduction (specialized for summation across unsigned long long types)
    __device__ __forceinline__ unsigned long long ReduceStep(
            unsigned long long  input,              ///< [in] Calling thread's input item.
            cub::Sum            /*reduction_op*/,   ///< [in] Binary reduction operator
            int                 last_lane,          ///< [in] Index of last lane in segment
            int                 offset)             ///< [in] Up-offset to pull from
    {
        unsigned long long output;
        int shfl_c = last_lane | SHFL_C;   // Shuffle control (mask and last_lane)

#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
            "{"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
            "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
            "  mov.b64 %0, {lo, hi};"
            "  @p add.u64 %0, %0, %1;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));
#else
        asm volatile(
        "{"
        "  .reg .u32 lo;"
        "  .reg .u32 hi;"
        "  .reg .pred p;"
        "  mov.b64 {lo, hi}, %1;"
        "  shfl.down.b32 lo|p, lo, %2, %3;"
        "  shfl.down.b32 hi|p, hi, %2, %3;"
        "  mov.b64 %0, {lo, hi};"
        "  @p add.u64 %0, %0, %1;"
        "}"
        : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c));
#endif

        return output;
    }


    /// Reduction (specialized for summation across long long types)
    __device__ __forceinline__ long long ReduceStep(
            long long           input,              ///< [in] Calling thread's input item.
            cub::Sum            /*reduction_op*/,   ///< [in] Binary reduction operator
            int                 last_lane,          ///< [in] Index of last lane in segment
            int                 offset)             ///< [in] Up-offset to pull from
    {
        long long output;
        int shfl_c = last_lane | SHFL_C;   // Shuffle control (mask and last_lane)

        // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
            "{"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
            "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
            "  mov.b64 %0, {lo, hi};"
            "  @p add.s64 %0, %0, %1;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));
#else
        asm volatile(
        "{"
        "  .reg .u32 lo;"
        "  .reg .u32 hi;"
        "  .reg .pred p;"
        "  mov.b64 {lo, hi}, %1;"
        "  shfl.down.b32 lo|p, lo, %2, %3;"
        "  shfl.down.b32 hi|p, hi, %2, %3;"
        "  mov.b64 %0, {lo, hi};"
        "  @p add.s64 %0, %0, %1;"
        "}"
        : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c));
#endif

        return output;
    }


    /// Reduction (specialized for summation across double types)
    __device__ __forceinline__ double ReduceStep(
            double              input,              ///< [in] Calling thread's input item.
            cub::Sum            /*reduction_op*/,   ///< [in] Binary reduction operator
            int                 last_lane,          ///< [in] Index of last lane in segment
            int                 offset)             ///< [in] Up-offset to pull from
    {
        double output;
        int shfl_c = last_lane | SHFL_C;   // Shuffle control (mask and last_lane)

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
            "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
            "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
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
        "  shfl.down.b32 lo|p, lo, %2, %3;"
        "  shfl.down.b32 hi|p, hi, %2, %3;"
        "  mov.b64 r0, {lo, hi};"
        "  @p add.f64 %0, %0, r0;"
        "}"
        : "=d"(output) : "d"(input), "r"(offset), "r"(shfl_c));
#endif

        return output;
    }




    /// Reduction step (generic)
    template <typename _T, typename ReductionOp>
    __device__ __forceinline__ _T ReduceStep(
            _T                  input,              ///< [in] Calling thread's input item.
            ReductionOp         reduction_op,       ///< [in] Binary reduction operator
            int                 last_lane,          ///< [in] Index of last lane in segment
            int                 offset)             ///< [in] Up-offset to pull from
    {
        _T output = input;

        _T temp = ShuffleDown<LOGICAL_WARP_THREADS>(output, offset, last_lane, member_mask);

        // Perform reduction op if valid
        if (offset + lane_id <= last_lane)
            output = reduction_op(input, temp);

        return output;
    }


    /// Reduction step (specialized for small unsigned integers size 32b or less)
    template <typename _T, typename ReductionOp>
    __device__ __forceinline__ _T ReduceStep(
            _T              input,                  ///< [in] Calling thread's input item.
            ReductionOp     reduction_op,           ///< [in] Binary reduction operator
            int             last_lane,              ///< [in] Index of last lane in segment
            int             offset,                 ///< [in] Up-offset to pull from
            Int2Type<true>  /*is_small_unsigned*/)  ///< [in] Marker type indicating whether T is a small unsigned integer
    {
        return ReduceStep(input, reduction_op, last_lane, offset);
    }


    /// Reduction step (specialized for types other than small unsigned integers size 32b or less)
    template <typename _T, typename ReductionOp>
    __device__ __forceinline__ _T ReduceStep(
            _T              input,                  ///< [in] Calling thread's input item.
            ReductionOp     reduction_op,           ///< [in] Binary reduction operator
            int             last_lane,              ///< [in] Index of last lane in segment
            int             offset,                 ///< [in] Up-offset to pull from
            Int2Type<false> /*is_small_unsigned*/)  ///< [in] Marker type indicating whether T is a small unsigned integer
    {
        return ReduceStep(input, reduction_op, last_lane, offset);
    }


    //---------------------------------------------------------------------
    // Templated inclusive scan iteration
    //---------------------------------------------------------------------

    template <typename ReductionOp, int STEP>
    __device__ __forceinline__ void ReduceStep(
            T&              input,              ///< [in] Calling thread's input item.
            ReductionOp     reduction_op,       ///< [in] Binary reduction operator
            int             last_lane,          ///< [in] Index of last lane in segment
            Int2Type<STEP>  /*step*/)
    {
        input = ReduceStep(input, reduction_op, last_lane, 1 << STEP, Int2Type<IsInteger<T>::IS_SMALL_UNSIGNED>());

        ReduceStep(input, reduction_op, last_lane, Int2Type<STEP + 1>());
    }

    template <typename ReductionOp>
    __device__ __forceinline__ void ReduceStep(
            T&              /*input*/,              ///< [in] Calling thread's input item.
            ReductionOp     /*reduction_op*/,       ///< [in] Binary reduction operator
            int             /*last_lane*/,          ///< [in] Index of last lane in segment
            Int2Type<STEPS> /*step*/)
    {}


    //---------------------------------------------------------------------
    // Reduction operations
    //---------------------------------------------------------------------

    /// Reduction
    template <
            bool            ALL_LANES_VALID,        ///< Whether all lanes in each warp are contributing a valid fold of items
            typename        ReductionOp>
    __device__ __forceinline__ T Reduce(
            T               input,                  ///< [in] Calling thread's input
            int             valid_items,            ///< [in] Total number of valid items across the logical warp
            ReductionOp     reduction_op)           ///< [in] Binary reduction operator
    {
        int last_lane = (ALL_LANES_VALID) ?
                        LOGICAL_WARP_THREADS - 1 :
                        valid_items - 1;

        T output = input;

//        // Iterate reduction steps
//        #pragma unroll
//        for (int STEP = 0; STEP < STEPS; STEP++)
//        {
//            output = ReduceStep(output, reduction_op, last_lane, 1 << STEP, Int2Type<IsInteger<T>::IS_SMALL_UNSIGNED>());
//        }

        // Template-iterate reduction steps
        ReduceStep(output, reduction_op, last_lane, Int2Type<0>());

        return output;
    }


    /// Segmented reduction
    template <
            bool            HEAD_SEGMENTED,     ///< Whether flags indicate a segment-head or a segment-tail
            typename        FlagT,
            typename        ReductionOp>
    __device__ __forceinline__ T SegmentedReduce(
            T               input,              ///< [in] Calling thread's input
            FlagT           flag,               ///< [in] Whether or not the current lane is a segment head/tail
            ReductionOp     reduction_op)       ///< [in] Binary reduction operator
    {
        // Get the start flags for each thread in the warp.
        int warp_flags = __ballot_sync(member_mask, flag);

        // Convert to tail-segmented
        if (HEAD_SEGMENTED)
            warp_flags >>= 1;

        // Mask out the bits below the current thread
        warp_flags &= LaneMaskGe();

        // Mask of physical lanes outside the logical warp and convert to logical lanemask
        if (!IS_ARCH_WARP)
        {
            warp_flags = (warp_flags & member_mask) >> (warp_id * LOGICAL_WARP_THREADS);
        }

        // Mask in the last lane of logical warp
        warp_flags |= 1u << (LOGICAL_WARP_THREADS - 1);

        // Find the next set flag
        int last_lane = __clz(__brev(warp_flags));

        T output = input;

//        // Iterate reduction steps
//        #pragma unroll
//        for (int STEP = 0; STEP < STEPS; STEP++)
//        {
//            output = ReduceStep(output, reduction_op, last_lane, 1 << STEP, Int2Type<IsInteger<T>::IS_SMALL_UNSIGNED>());
//        }

        // Template-iterate reduction steps
        ReduceStep(output, reduction_op, last_lane, Int2Type<0>());

        return output;
    }
};


}

#endif //CUBSCAN_WARP_REDUCE_SHFL_CUH
