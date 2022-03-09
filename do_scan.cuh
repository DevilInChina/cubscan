//
// Created by yxgao on 2022/3/7.
//
#include "block_scan.cuh"
#include "warp_reduce_shfl.cuh"
#include "scan_status.cuh"

#ifndef CUBSCAN_DO_SCAN_CUH
#define CUBSCAN_DO_SCAN_CUH


namespace scan {


    template<typename T, typename ScanTileStatusT = ScanTileState<T>>
    size_t get_temp_storage_size(int num_items) {
        size_t ret;
        int num_tiles = (num_items + SCAN_TILE_SIZE - 1) / SCAN_TILE_SIZE;
        ScanTileStatusT::AllocationSize(num_tiles, ret);
        return ret;
    }

    template<typename T, typename ScanTileStatusT = ScanTileState<T>>
    __global__ void scan_do_init(void *temp_storage, int num_tiles, size_t temp_size) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        ScanTileStatusT ss;
        ss.Init(num_tiles, temp_storage, temp_size);

        ss.InitializeStatus(num_tiles);
    }

    template<typename T, typename ScanTileStatusT = ScanTileState<T>>
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
        get_block_aggregate(T block_aggregate, unsigned int tile_idx,
                            T shared_info[4]) {
            if (threadIdx.x == 0) {
                shared_info[0] = block_aggregate; /// shared
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

                shared_info[1] = exclusive_prefix;
                shared_info[2] = inclusive_prefix;
            }

            // Return exclusive_prefix
            return exclusive_prefix;
        }

        __device__ __forceinline__ void ExclusiveScan(T d_in, T &exclusive_output) {
            unsigned int tile_idx = blockIdx.x;
            T block_aggregate;


            /// load

            scan::BlockScanWarpScans<T, SCAN_BLOCK_SIZE, 1, 1, CUB_PTX_ARCH>().ExclusiveScan(d_in, exclusive_output,
                                                                                             block_aggregate);
            __shared__ T data[4];
            if (blockIdx.x == 0) {
                if (threadIdx.x == 0) {
                    scanTileStatusT.SetInclusive(0, block_aggregate);
                    data[3] = 0;
                }

            } else if (threadIdx.x < SCAN_WARP_SIZE) { /// warp id 0
                T block_prefix = get_block_aggregate(block_aggregate, tile_idx, data);
                if (threadIdx.x % SCAN_WARP_SIZE == 0) {/// lane id 0
                    // Share the prefix with all threads
                    data[3] = block_prefix;
                    exclusive_output = block_prefix;                // The block prefix is the exclusive output for tid0
                }
            }
            __syncthreads();

            T block_prefix = data[3];
            if (threadIdx.x > 0) {
                exclusive_output = block_prefix + exclusive_output;
            }
            /// store
        }
    };

    template<typename T>
    __global__ void
    scan_do_kernel(void *temp_storage, T *d_in, T *d_out, int num_items, int num_tiles, size_t temp_size) {
        int item_idx = blockDim.x * blockIdx.x + threadIdx.x;
        T d_inputs[SCAN_ITEM_PER_BLOCK];
        int begin_idx = SCAN_ITEM_PER_BLOCK * item_idx;
#pragma unroll
        for (int i = 0; i < SCAN_ITEM_PER_BLOCK; ++i) {
            if (i + begin_idx < num_items) {
                d_inputs[i] = d_in[begin_idx + i];
            } else {
                d_inputs[i] = 0;
            }
        }

        T d_input = 0, d_output;
#pragma unroll
        for (int i = 0; i < SCAN_ITEM_PER_BLOCK; ++i) {
            d_input += d_inputs[i];
        }

        scan_block<T> ss(temp_storage, num_tiles, temp_size);
        ss.ExclusiveScan(d_input, d_output);


        d_input = d_inputs[0];
        d_inputs[0] = d_output;
#pragma unroll
        for (int i = 1; i < SCAN_ITEM_PER_BLOCK; ++i) {
            T new_val = d_inputs[i];
            d_inputs[i] = d_input + d_inputs[i - 1];
            d_input = new_val;
        }
#pragma unroll
        for (int i = 0; i < SCAN_ITEM_PER_BLOCK; ++i) {
            if (i + begin_idx < num_items)
                d_out[i + begin_idx] = d_inputs[i];
        }
    }


    template<typename T>
    void ExclusiveScan(void *temp_storage, T *d_in, T *d_out, int num_items, cudaStream_t cudaStream) {
        unsigned int num_tiles = (num_items + SCAN_TILE_SIZE - 1) / SCAN_TILE_SIZE;
        unsigned int init_block_size = (num_tiles + 31) / 32;
        size_t temp_size = get_temp_storage_size<T>(num_items);
        scan_do_init<T><<<init_block_size, 32, 0, cudaStream>>>(temp_storage, num_tiles, temp_size);

        scan_do_kernel<T> <<<num_tiles, SCAN_BLOCK_SIZE, 0, cudaStream>>>(temp_storage, d_in, d_out,
                                                                          num_items, num_tiles, temp_size);


    }
}


#endif //CUBSCAN_DO_SCAN_CUH
