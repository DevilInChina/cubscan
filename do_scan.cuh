//
// Created by yxgao on 2022/3/7.
//
#include "block_scan.cuh"
#include "warp_reduce_shfl.cuh"
#include "scan_status.cuh"

#ifndef CUBSCAN_DO_SCAN_CUH
#define CUBSCAN_DO_SCAN_CUH


namespace scan {


    template<typename T, typename ScanTileStatusT = CUB_NS_QUALIFIER::ScanTileState<T>>
    size_t get_temp_storage_size(int num_items) {
        size_t ret;
        int num_tiles = (num_items + SCAN_TILE_SIZE - 1) / SCAN_TILE_SIZE;
        ScanTileStatusT::AllocationSize(num_tiles, ret);
        size_t temp_storage_bytes;

        size_t allocation_sizes[1] = {ret};
        void *allocations[1] = {};
        CUB_NS_QUALIFIER::AliasTemporaries(nullptr, temp_storage_bytes, allocations, allocation_sizes);
        return temp_storage_bytes;
    }

    template<typename T, typename ScanTileStatusT = CUB_NS_QUALIFIER::ScanTileState<T>>
    __global__ void scan_do_init(void *temp_storage, int num_tiles, size_t temp_size) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        ScanTileStatusT ss;
        ss.Init(num_tiles, temp_storage, temp_size);

        ss.InitializeStatus(num_tiles);
    }

    template<typename T, typename ScanTileStatusT = CUB_NS_QUALIFIER::ScanTileState<T>>
    struct scan_block {
        ScanTileStatusT scanTileStatusT;
        using status_words = typename ScanTileStatusT::StatusWord;

        __device__ __forceinline__ scan_block(void *temp_storage, int num_tiles, size_t temp_size) {
            scanTileStatusT.Init(num_tiles, temp_storage, temp_size);
        }

        __device__ __forceinline__
        void ProcessWindow(
                int predecessor_idx,        ///< Preceding tile index to inspect
                status_words &predecessor_status,    ///< [out] Preceding tile status
                T &window_aggregate)      ///< [out] Relevant partial reduction from this window of preceding tiles
        {
            T value;

            scanTileStatusT.WaitForValid(predecessor_idx, predecessor_status, value);

            // Perform a segmented reduction to get the prefix for the current window.
            // Use the swizzled scan operator because we are now scanning *down* towards thread0.

            int tail_flag = (predecessor_status == SCAN_TILE_INCLUSIVE);

            window_aggregate = scan::WarpReduceShfl<T, SCAN_WARP_SIZE, CUB_PTX_ARCH>().template SegmentedReduce<false>(
                    value, tail_flag,
                    scan::cub::Sum());
        }

        __device__ __forceinline__ T
        get_block_aggregate(T block_aggregate, unsigned int tile_idx) {
            if (threadIdx.x == 0) {
                scanTileStatusT.SetPartial(tile_idx, block_aggregate);
            }

            scan::cub::Sum scan_op;
            int predecessor_idx = tile_idx - threadIdx.x - 1;
            status_words predecessor_status;
            T window_aggregate;

            ProcessWindow(predecessor_idx, predecessor_status, window_aggregate);
            T exclusive_prefix = window_aggregate;
            T inclusive_prefix;
            while (__all_sync(0xffffffff, predecessor_status != SCAN_TILE_INCLUSIVE)) {
                predecessor_idx -= SCAN_WARP_SIZE;

                // Update exclusive tile prefix with the window prefix
                ProcessWindow(predecessor_idx, predecessor_status, window_aggregate);
                exclusive_prefix = scan_op(window_aggregate, exclusive_prefix);

            }

            if (threadIdx.x == 0) {
                inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
                scanTileStatusT.SetInclusive(tile_idx, inclusive_prefix);
            }

            // Return exclusive_prefix
            return exclusive_prefix;
        }

        template<
                int LENGTH,
                typename ReductionOp>
        __device__ __forceinline__ T ThreadReduce(
                T *input,                  ///< [in] Input array
                ReductionOp reduction_op,           ///< [in] Binary reduction operator
                T prefix,                 ///< [in] Prefix to seed reduction with
                Int2Type<LENGTH>    /*length*/) {
            T retval = prefix;

#pragma unroll
            for (int i = 0; i < LENGTH; ++i)
                retval = reduction_op(retval, input[i]);

            return retval;

        }

        template<
                int LENGTH,
                typename ScanOp>
        __device__ __forceinline__ T ThreadScanExclusive(
                T inclusive,
                T exclusive,
                T *input,                 ///< [in] Input array
                T *output,                ///< [out] Output array (may be aliased to \p input)
                ScanOp scan_op,                ///< [in] Binary scan operator
                Int2Type<LENGTH>    /*length*/) {
#pragma unroll
            for (int i = 0; i < LENGTH; ++i) {
                inclusive = scan_op(exclusive, input[i]);
                output[i] = exclusive;
                exclusive = inclusive;
            }

            return inclusive;
        }

        using OutputT = T;
        enum {
            ITEMS_PER_THREAD = SCAN_ITEM_PER_BLOCK
        };
        using ScanOpT = CUB_NS_QUALIFIER::Sum;
        typedef typename CUB_NS_QUALIFIER::BlockScan<T,SCAN_BLOCK_SIZE,CUB_NS_QUALIFIER::BLOCK_SCAN_WARP_SCANS>  BlockScanT;
        /**
         * Exclusive scan specialization (first tile)
         */
         struct ScanTileT {
             typename BlockScanT::TempStorage &temp_storage;
             __device__ __forceinline__
             void ScanTile(
                     OutputT             (&items)[ITEMS_PER_THREAD],
                     OutputT init_value,
                     ScanOpT scan_op,
                     OutputT &block_aggregate,
                     Int2Type<false>     /*is_inclusive*/) {
                 BlockScanT(temp_storage.scan_storage.scan).ExclusiveScan(items, items, init_value, scan_op,
                                                                          block_aggregate);
                 block_aggregate = scan_op(init_value, block_aggregate);
             }


             /**
              * Inclusive scan specialization (first tile)
              */
             __device__ __forceinline__
             void ScanTile(
                     OutputT             (&items)[ITEMS_PER_THREAD],
                     ScanOpT scan_op,
                     OutputT &block_aggregate,
                     Int2Type<true>      /*is_inclusive*/) {
                 BlockScanT(temp_storage.scan_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate);
             }


             /**
              * Exclusive scan specialization (subsequent tiles)
              */
             template<typename PrefixCallback>
             __device__ __forceinline__
             void ScanTile(
                     OutputT             (&items)[ITEMS_PER_THREAD],
                     ScanOpT scan_op,
                     PrefixCallback &prefix_op,
                     Int2Type<false>     /*is_inclusive*/) {
                 BlockScanT(temp_storage.scan_storage.scan).ExclusiveScan(items, items, scan_op, prefix_op);
             }


             /**
              * Inclusive scan specialization (subsequent tiles)
              */
             template<typename PrefixCallback>
             __device__ __forceinline__
             void ScanTile(
                     OutputT             (&items)[ITEMS_PER_THREAD],
                     ScanOpT scan_op,
                     PrefixCallback &prefix_op,
                     Int2Type<true>      /*is_inclusive*/) {
                 BlockScanT(temp_storage.scan_storage.scan).InclusiveScan(items, items, scan_op, prefix_op);
             }
         };

        __device__ __forceinline__ void
        ExclusiveScan(T d_inputs[SCAN_ITEM_PER_BLOCK], T d_outpus[SCAN_ITEM_PER_BLOCK]) {
            unsigned int tile_idx = blockIdx.x;
            T block_aggregate;

            __shared__ typename BlockScanT::TempStorage tempStorage;
            BlockScanT ss(tempStorage);
            CUB_NS_QUALIFIER::Sum scan_op;
            Int2Type<SCAN_ITEM_PER_BLOCK> teet;
            T d_in = ThreadReduce<SCAN_ITEM_PER_BLOCK, CUB_NS_QUALIFIER::Sum>(d_inputs, scan_op, 0,
                                                                              teet), exclusive_output;

            /// load
            ss.ExclusiveSum(d_in, exclusive_output, block_aggregate);
            //scan::BlockScanWarpScans<T, SCAN_BLOCK_SIZE, 1, 1, CUB_PTX_ARCH>().ExclusiveScan(d_in, exclusive_output,block_aggregate);
            __shared__ T data_pref;
            if (blockIdx.x == 0) {
                if (threadIdx.x == 0) {
                    scanTileStatusT.SetInclusive(0, block_aggregate);
                    data_pref = 0;
                }

            } else if (threadIdx.x < SCAN_WARP_SIZE) { /// warp id 0
                T block_prefix = get_block_aggregate(block_aggregate, tile_idx);
                if (threadIdx.x % SCAN_WARP_SIZE == 0) {/// lane id 0
                    // Share the prefix with all threads
                    data_pref = block_prefix;
                    exclusive_output = block_prefix;                // The block prefix is the exclusive output for tid0
                }
            }
            __syncthreads();

            T block_prefix = data_pref;
            if (threadIdx.x > 0) {
                exclusive_output = block_prefix + exclusive_output;
            }

            __syncthreads();
            ThreadScanExclusive<SCAN_ITEM_PER_BLOCK, CUB_NS_QUALIFIER::Sum>(d_in, exclusive_output, d_inputs, d_outpus,
                                                                            scan_op, Int2Type<SCAN_ITEM_PER_BLOCK>());

            /// store
        }
    };

    template<typename T>
    __global__ void
    scan_do_kernel(void *temp_storage, T *d_in, T *d_out, int num_items, int num_tiles, size_t temp_size) {
        int item_idx = blockDim.x * blockIdx.x + threadIdx.x;
        T d_inputs[SCAN_ITEM_PER_BLOCK];
        T d_outputs[SCAN_ITEM_PER_BLOCK];
        int begin_idx = SCAN_ITEM_PER_BLOCK * item_idx;
#pragma unroll
        for (int i = 0; i < SCAN_ITEM_PER_BLOCK; ++i) {
            if (i + begin_idx < num_items) {
                d_inputs[i] = d_in[begin_idx + i];
            } else {
                d_inputs[i] = 0;
            }
        }

        scan_block<T> ss(temp_storage, num_tiles, temp_size);
        ss.ExclusiveScan(d_inputs, d_outputs);

        if (blockIdx.x == num_tiles - 1)
#pragma unroll
            for (int i = 0; i < SCAN_ITEM_PER_BLOCK; ++i) {
                if (i + begin_idx < num_items)
                    d_out[i + begin_idx] = d_outputs[i];
            }
        else {
#pragma unroll
            for (int i = 0; i < SCAN_ITEM_PER_BLOCK; ++i) {
                d_out[i + begin_idx] = d_outputs[i];
            }
        }
    }


    template<typename T, typename ScanTileStatusT = CUB_NS_QUALIFIER::ScanTileState<T> >
    void ExclusiveScan(void *d_temp_storage, T *d_in, T *d_out, int num_items, cudaStream_t cudaStream) {
        unsigned int num_tiles = (num_items + SCAN_TILE_SIZE - 1) / SCAN_TILE_SIZE;
        unsigned int init_block_size = (num_tiles + 31) / 32;
        size_t temp_size = get_temp_storage_size<T>(num_items);
        size_t allocation_sizes[1];
        ScanTileStatusT::AllocationSize(num_tiles, allocation_sizes[0]);
        void *allocations[1] = {};
        size_t temp_storage_bytes;
        CUB_NS_QUALIFIER::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
        scan_do_init<T><<<init_block_size, 32, 0, cudaStream>>>(allocations[0], num_tiles, temp_size);

        scan_do_kernel<T> <<<num_tiles, SCAN_BLOCK_SIZE, 0, cudaStream>>>(allocations[0], d_in, d_out,
                                                                          num_items, num_tiles, temp_size);


    }
}


#endif //CUBSCAN_DO_SCAN_CUH
