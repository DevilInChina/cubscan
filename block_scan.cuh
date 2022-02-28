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
                linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z)),
                warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS), lane_id(LaneId()) {}

        template <typename ScanOp>
        __device__ __forceinline__ T ComputeWarpPrefix(
                ScanOp          scan_op,            ///< [in] Binary scan operator
                T               warp_aggregate,     ///< [in] <b>[<em>lane</em><sub>WARP_THREADS - 1</sub> only]</b> Warp-wide aggregate reduction of input items
                T               &block_aggregate)   ///< [out] Threadblock-wide aggregate reduction of input items
        {
            // Last lane in each warp shares its warp-aggregate
            if (lane_id == WARP_THREADS - 1)
                temp_storage.warp_aggregates[warp_id] = warp_aggregate;

            CTA_SYNC();


            // Accumulate block aggregates and save the one that is our warp's prefix
            T warp_prefix;
            block_aggregate = temp_storage.warp_aggregates[0];

            // Use template unrolling (since the PTX backend can't handle unrolling it for SM1x)
            ApplyWarpAggregates(warp_prefix, scan_op, block_aggregate, Int2Type<1>());
/*
        #pragma unroll
        for (int WARP = 1; WARP < WARPS; ++WARP)
        {
            if (warp_id == WARP)
                warp_prefix = block_aggregate;

            T addend = temp_storage.warp_aggregates[WARP];
            block_aggregate = scan_op(block_aggregate, addend);
        }
*/

            return warp_prefix;
        }
    };
}
#endif //CUBSCAN_BLOCK_SCAN_CUH
