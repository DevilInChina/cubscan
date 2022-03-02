//
// Created by yxgao on 2022/2/28.
//
#include "warp_scan_shfl.cuh"

#ifndef CUBSCAN_BLOCK_SCAN_CUH
#define CUBSCAN_BLOCK_SCAN_CUH
namespace scan {
    template<
            typename T,
            int BLOCK_DIM_X,    ///< The thread block length in threads along the X dimension
            int BLOCK_DIM_Y,    ///< The thread block length in threads along the Y dimension
            int BLOCK_DIM_Z,    ///< The thread block length in threads along the Z dimension
            int PTX_ARCH>       ///< The PTX compute capability for which to to specialize this collective
    struct BlockScanWarpScans {
        //---------------------------------------------------------------------
        // Types and constants
        //---------------------------------------------------------------------

        /// Constants
        enum {
            /// Number of warp threads
            WARP_THREADS = CUB_WARP_THREADS(PTX_ARCH),

            /// The thread block size in threads
            BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

            /// Number of active warps
            WARPS = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,
        };
        unsigned int linear_tid;
        unsigned int warp_id;
        unsigned int lane_id;

        __device__ __forceinline__ BlockScanWarpScans() :
                linear_tid(origin_cub::RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z)),
                warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS), lane_id(LaneId()) {}

        template<int WARP>
        __device__ __forceinline__ void ApplyWarpAggregates(
                T &warp_prefix,           ///< [out] The calling thread's partial reduction
                T &block_aggregate,   ///< [out] Threadblock-wide aggregate reduction of input items
                T *shared_warp_aggregate,
                Int2Type<WARP>  /*addend_warp*/) {
            if (warp_id == WARP)
                warp_prefix = block_aggregate;

            T addend = shared_warp_aggregate[WARP];
            block_aggregate = block_aggregate + addend;

            ApplyWarpAggregates(warp_prefix, block_aggregate, shared_warp_aggregate, Int2Type<WARP + 1>());
        }

        __device__ __forceinline__ void ApplyWarpAggregates(
                T &/*warp_prefix*/,       ///< [out] The calling thread's partial reduction
                T &/*block_aggregate*/,   ///< [out] Threadblock-wide aggregate reduction of input items
                T *shared_warp_aggregate,
                Int2Type<WARPS> /*addend_warp*/) {}

        __device__ __forceinline__ T ComputeWarpPrefix(
                T warp_aggregate,     ///< [in] <b>[<em>lane</em><sub>WARP_THREADS - 1</sub> only]</b> Warp-wide aggregate reduction of input items
                T &block_aggregate)   ///< [out] Threadblock-wide aggregate reduction of input items
        {
            // Last lane in each warp shares its warp-aggregate
            __shared__ T warp_aggregates[WARPS];
            if (lane_id == WARP_THREADS - 1)
                warp_aggregates[warp_id] = warp_aggregate;

            __syncthreads();


            // Accumulate block aggregates and save the one that is our warp's prefix
            T warp_prefix;
            block_aggregate = warp_aggregates[0];

            // Use template unrolling (since the PTX backend can't handle unrolling it for SM1x)
            ApplyWarpAggregates(warp_prefix, block_aggregate, warp_aggregates, Int2Type<1>());

            return warp_prefix;
        }

        __device__ __forceinline__ void ExclusiveScan(T input,              ///< [in] Calling thread's input item
                                                      T &exclusive_output,  ///< [out] Calling thread's output item (may be aliased to \p input)
                                                      T &block_aggregate) {
            T inclusive_output = 0;
            scan::warp_scan_shfl<T, WARP_THREADS, PTX_ARCH>().InclusiveScan(input, inclusive_output, exclusive_output);
            T warp_prefix = ComputeWarpPrefix(inclusive_output, block_aggregate);

            if (warp_id != 0)
            {
                exclusive_output = warp_prefix+exclusive_output;
            }
        }
        __device__ __forceinline__ void InclusiveScan(T input,              ///< [in] Calling thread's input item
                                                      T &inclusive_output,  ///< [out] Calling thread's output item (may be aliased to \p input)
                                                      T &block_aggregate) {
            T exclusive_output = 0;
            scan::warp_scan_shfl<T, WARP_THREADS, PTX_ARCH>().InclusiveScan(input, inclusive_output, exclusive_output);
            T warp_prefix = ComputeWarpPrefix(inclusive_output, block_aggregate);

            if (warp_id != 0)
            {
                inclusive_output = warp_prefix+inclusive_output;
            }
        }
    };
}
#endif //CUBSCAN_BLOCK_SCAN_CUH
