//
// Created by yxgao on 2022/3/7.
//
#include "block_scan.cuh"
#include "warp_reduce_shfl.cuh"
#include "scan_status.cuh"
#ifndef CUBSCAN_DO_SCAN_CUH
#define CUBSCAN_DO_SCAN_CUH


namespace scan {
    template<typename T>
    struct ScanTile {
        T prefix;
        ScanTileStatus status;
    };


    template<typename T>
    unsigned int get_temp_storage_size(int num_items) {
        int num_tiles = (num_items + SCAN_TILE_SIZE - 1) / SCAN_TILE_SIZE;
        return sizeof(ScanTile<T>) * (num_tiles + SCAN_TILE_STATUS_PADDING);
    }

    template<typename T>
    __global__ void scan_do_init(ScanTile<T> *temp_storage, int num_tiles) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        ScanTile<T> *scanTiles = temp_storage;
        ScanTile<T> temp;
        temp.prefix = 0;

        if (tid < SCAN_TILE_STATUS_PADDING) {
            temp.status = SCAN_TILE_OOB;
        } else temp.status = SCAN_TILE_INVALID;
        if (tid < SCAN_TILE_STATUS_PADDING + num_tiles) {
            scanTiles[tid] = temp;
        }
    }

    template<typename T>
    __device__ __forceinline__ void setPartial(ScanTile<T> *scanTiles, T value, unsigned int tile_id) {
        ScanTile<T> temp;
        temp.prefix = value;
        temp.status = SCAN_TILE_PARTIAL;
        //__threadfence();
        scanTiles[tile_id + SCAN_TILE_STATUS_PADDING] = temp;
    }

    template<typename T>
    __device__ __forceinline__ void setInclusive(ScanTile<T> *scanTiles, T value, unsigned int tile_id) {
        ScanTile<T> temp;
        temp.prefix = value;
        temp.status = SCAN_TILE_INCLUSIVE;
        //__threadfence();
        scanTiles[tile_id + SCAN_TILE_STATUS_PADDING] = temp;
    }

    template<typename T>
    __device__ __forceinline__ void WaitForValid(
            int tile_idx,
            ScanTile<T> *d_tile_descriptors,
            ScanTileStatus &status,
            T &value) {
        ScanTile<T> tile_descriptor;
        do {
            __threadfence(); // prevent hoisting loads from loop
            ScanTile<T> *alias = (d_tile_descriptors + SCAN_TILE_STATUS_PADDING + tile_idx);
            tile_descriptor = *alias;

        } while (__any_sync(0xffffffff, tile_descriptor.status == SCAN_TILE_INVALID));
        status = tile_descriptor.status;
        value = tile_descriptor.prefix;
    }


    template<typename T>
    __device__ __forceinline__
    void ProcessWindow(
            int predecessor_idx,        ///< Preceding tile index to inspect
            ScanTile<T> *d_tile_descriptors,
            ScanTileStatus &predecessor_status,    ///< [out] Preceding tile status
            T &window_aggregate)      ///< [out] Relevant partial reduction from this window of preceding tiles
    {
        T value;

        WaitForValid(predecessor_idx, d_tile_descriptors, predecessor_status, value);

        // Perform a segmented reduction to get the prefix for the current window.
        // Use the swizzled scan operator because we are now scanning *down* towards thread0.

        int tail_flag = (predecessor_status == SCAN_TILE_INCLUSIVE);

        window_aggregate = scan::WarpReduceShfl<T, SCAN_WARP_SIZE, CUB_PTX_ARCH>().template SegmentedReduce<false>(
                value, tail_flag,
                scan::cub::Sum());
    }

    template<typename T>
    __device__ __forceinline__ T
    get_block_aggregate(T block_aggregate, ScanTile<T> *d_tile_descriptors, unsigned int tile_idx, T shared_info[4]) {
        if (threadIdx.x == 0) {
            shared_info[0] = block_aggregate; /// shared
            setPartial(d_tile_descriptors, block_aggregate, tile_idx);
        }

        scan::cub::Sum scan_op;
        int predecessor_idx = tile_idx - threadIdx.x - 1;
        ScanTileStatus predecessor_status;
        T window_aggregate;

        ProcessWindow(predecessor_idx, d_tile_descriptors, predecessor_status, window_aggregate);
        T exclusive_prefix = window_aggregate;
        T inclusive_prefix;
        while (__all_sync(0xffffffff, predecessor_status != SCAN_TILE_INCLUSIVE)) {
            predecessor_idx -= SCAN_WARP_SIZE;

            // Update exclusive tile prefix with the window prefix
            ProcessWindow(predecessor_idx, d_tile_descriptors, predecessor_status, window_aggregate);
            exclusive_prefix = scan_op(window_aggregate, exclusive_prefix);

        }

        if (threadIdx.x == 0) {
            inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
            setInclusive(d_tile_descriptors, inclusive_prefix, tile_idx);

            shared_info[1] = exclusive_prefix;
            shared_info[2] = inclusive_prefix;
        }

        // Return exclusive_prefix
        return exclusive_prefix;
    }

    template<typename T>
    __device__ __forceinline__ void ExclusiveScan(ScanTile<T> *temp_storage, T d_in, T &exclusive_output) {
        unsigned int tile_idx = blockIdx.x;
        T block_aggregate;


        /// load

        scan::BlockScanWarpScans<T, SCAN_BLOCK_SIZE, 1, 1, CUB_PTX_ARCH>().ExclusiveScan(d_in, exclusive_output,
                                                                                         block_aggregate);
        __shared__ T data[4];
        if (blockIdx.x == 0) {
            if(threadIdx.x == 0) {
                setInclusive(temp_storage, block_aggregate, 0);
                data[3] = 0;
            }

        } else if (threadIdx.x < SCAN_WARP_SIZE) { /// warp id 0
            T block_prefix = get_block_aggregate(block_aggregate, temp_storage, tile_idx, data);
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

    template<typename T>
    __global__ void scan_do_kernel(ScanTile<T> *temp_storage, T *d_in, T *d_out, int num_items) {
        int item_idx = blockDim.x * blockIdx.x + threadIdx.x;

        T d_input = item_idx < num_items ? d_in[item_idx] : 0;
        T d_output = 0;
        ExclusiveScan(temp_storage, d_input, d_output);
        if (item_idx < num_items) {
            d_out[item_idx] = d_output;
        }
    }

    template<typename T>
    void ExclusiveScan(void *temp_storage, T *d_in, T *d_out, int num_items, cudaStream_t cudaStream) {
        unsigned int num_tiles = (num_items + SCAN_TILE_SIZE - 1) / SCAN_TILE_SIZE;
        unsigned int init_block_size = (num_tiles + 31 + SCAN_TILE_STATUS_PADDING) / 32;

        scan_do_init < T ><<<init_block_size, 32, 0, cudaStream>>>((ScanTile<T> *) temp_storage, num_tiles);

        scan_do_kernel < T > <<<num_tiles, SCAN_BLOCK_SIZE, 0, cudaStream>>>((ScanTile<T> *) temp_storage, d_in, d_out,
                num_items);


    }
}


#endif //CUBSCAN_DO_SCAN_CUH
