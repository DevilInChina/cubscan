//
// Created by yxgao on 2022/3/7.
//
#include "block_scan.cuh"
#include "warp_reduce_shfl.cuh"

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

    template<typename T, typename ScanTileStatusT>
    __global__ void scan_do_init(void *temp_storage, int num_tiles, size_t temp_size) {
        ScanTileStatusT ss;
        ss.Init(num_tiles, temp_storage, temp_size);

        ss.InitializeStatus(num_tiles);
    }

    template<typename T, typename ScanTileStatusT>
    struct scan_block {
        ScanTileStatusT scanTileStatusT;
        using status_words = typename ScanTileStatusT::StatusWord;

        using ScanOpT = CUB_NS_QUALIFIER::Sum;
        typedef typename CUB_NS_QUALIFIER::BlockScan<T, SCAN_BLOCK_SIZE, CUB_NS_QUALIFIER::BLOCK_SCAN_WARP_SCANS> BlockScanT;

        using OutputT = T;
        using TilePrefixCallbackOpT = CUB_NS_QUALIFIER::TilePrefixCallbackOp<
                OutputT,
                ScanOpT,
                ScanTileStatusT>;

        enum {
            ITEMS_PER_THREAD = SCAN_ITEM_PER_BLOCK
        };

        typedef CUB_NS_QUALIFIER::BlockLoad<
                OutputT,
                SCAN_BLOCK_SIZE,
                SCAN_ITEM_PER_BLOCK,
                CUB_NS_QUALIFIER::BLOCK_LOAD_DIRECT>
                BlockLoadT;

        // Parameterized BlockStore type
        typedef CUB_NS_QUALIFIER::BlockStore<
                OutputT,
                SCAN_BLOCK_SIZE,
                SCAN_ITEM_PER_BLOCK,
                CUB_NS_QUALIFIER::BLOCK_STORE_WARP_TRANSPOSE>
                BlockStoreT;
        union _TempStorage {
            typename BlockLoadT::TempStorage load;       // Smem needed for tile loading
            typename BlockStoreT::TempStorage store;      // Smem needed for tile storing

            struct ScanStorage {
                typename TilePrefixCallbackOpT::TempStorage prefix;     // Smem needed for cooperative prefix callback
                typename BlockScanT::TempStorage scan;       // Smem needed for tile scanning
            } scan_storage;
        };

        // Alias wrapper allowing storage to be unioned
        struct TempStorage : CUB_NS_QUALIFIER::Uninitialized<_TempStorage> {
        };


        //---------------------------------------------------------------------
        // Per-thread fields
        //---------------------------------------------------------------------

        _TempStorage &temp_storage;       ///< Reference to temp_storage

        __device__ __forceinline__ _TempStorage &PrivateStorage() {
            __shared__ _TempStorage private_storage;
            return private_storage;
        }

        __device__ __forceinline__ scan_block(
                void *temp_storage, int num_tiles, size_t temp_size) : temp_storage(PrivateStorage()) {
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

        /**
         * Exclusive scan specialization (first tile)
         */

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

        __device__ __forceinline__ void
        ExclusiveScan(OutputT             (&d_inputs)[ITEMS_PER_THREAD]) {
            unsigned int tile_idx = blockIdx.x;
            CUB_NS_QUALIFIER::Sum scan_op;
            // Perform tile scan
            if (tile_idx == 0) {
                // Scan first tile
                OutputT block_aggregate;
                ScanTile(d_inputs, scan_op, block_aggregate, Int2Type<false>());
                if ((threadIdx.x == 0))
                    scanTileStatusT.SetInclusive(0, block_aggregate);
            } else {
                // Scan non-first tile
                TilePrefixCallbackOpT prefix_op(scanTileStatusT, temp_storage.scan_storage.prefix, scan_op, tile_idx);
                ScanTile(d_inputs, scan_op, prefix_op, Int2Type<false>());
            }
            /// load


            /// store
        }

        __device__ __forceinline__
        void ExclusiveScan_AgentScan(T *d_in, T *d_out, int num_items, int num_tiles) {
            using InitValueT = CUB_NS_QUALIFIER::detail::InputValue<T>;
            using RealInitValueT = typename InitValueT::value_type;
            typedef typename CUB_NS_QUALIFIER::AgentScan<
                    typename CUB_NS_QUALIFIER::DeviceScanPolicy<T>::Policy520::ScanPolicyT,
                    T *,
                    T *,
                    CUB_NS_QUALIFIER::Sum,
                    RealInitValueT,
                    int> AgentScanT;
            __shared__ typename AgentScanT::TempStorage temp_storage;
            T init_value = 0;
            RealInitValueT real_init_value = CUB_NS_QUALIFIER::detail::InputValue<T>(init_value);
            CUB_NS_QUALIFIER::Sum scan_op;
            AgentScanT temp(temp_storage, d_in, d_out, scan_op, real_init_value);
            temp.ConsumeRange(
                    num_items,
                    scanTileStatusT,
                    0);

        };

        __device__ __forceinline__
        void ExclusiveScan(T *d_in, T *d_out, int num_items, int num_tiles) { /// consume range



            int item_idx = blockDim.x * blockIdx.x + threadIdx.x;
            T items[SCAN_ITEM_PER_BLOCK];
            int tile_offset = SCAN_TILE_SIZE * blockIdx.x;

            int num_remaining = num_items - tile_offset;          // Remaining items (including this tile)
            if (blockIdx.x == num_tiles - 1) {
                // Fill last element with the first element because collectives are
                // not suffix guarded.
                BlockLoadT(temp_storage.load)
                        .Load(d_in + tile_offset,
                              items,
                              num_remaining,
                              *(d_in + tile_offset));
            } else {
                BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
            }
            __syncthreads();
            ExclusiveScan(items);
            __syncthreads();


            // Store items
            if (blockIdx.x == num_tiles - 1)
                BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items, num_remaining);
            else
                BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
        }

    };

    template<typename T, typename ScanTileStatusT>
    __global__ void
    scan_do_kernel(void *d_temp_storage, T *d_in, T *d_out, int num_items, int num_tiles, size_t temp_size) {

        scan_block<T, ScanTileStatusT> ss(d_temp_storage, num_tiles, temp_size);
        ss.ExclusiveScan(d_in, d_out, num_items, num_tiles);
        /*
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

        scan_block<T, ScanTileStatusT> ss(d_temp_storage, num_tiles, temp_size);
        ss.ExclusiveScan(d_inputs);

        if (blockIdx.x == num_tiles - 1)
#pragma unroll
            for (int i = 0; i < SCAN_ITEM_PER_BLOCK; ++i) {
                if (i + begin_idx < num_items)
                    d_out[i + begin_idx] = d_inputs[i];
            }
        else {
#pragma unroll
            for (int i = 0; i < SCAN_ITEM_PER_BLOCK; ++i) {
                d_out[i + begin_idx] = d_inputs[i];
            }
        }*/
    }


    template<typename T>
    void ExclusiveScan(void *d_temp_storage, T *d_in, T *d_out, int num_items, cudaStream_t cudaStream) {
        unsigned int num_tiles = (num_items + SCAN_TILE_SIZE - 1) / SCAN_TILE_SIZE;
        unsigned int init_block_size = (num_tiles + 31) / 32;
        size_t temp_size = get_temp_storage_size<T>(num_items);
        size_t allocation_sizes[1];
        using ScanTileStatusT = CUB_NS_QUALIFIER::ScanTileState<T>;
        ScanTileStatusT::AllocationSize(num_tiles, allocation_sizes[0]);
        void *allocations[1] = {};
        size_t temp_storage_bytes;
        CUB_NS_QUALIFIER::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
        scan_do_init<T, ScanTileStatusT><<<init_block_size, 32, 0, cudaStream>>>(allocations[0], num_tiles, temp_size);

        scan_do_kernel<T, ScanTileStatusT> <<<num_tiles, SCAN_BLOCK_SIZE, 0, cudaStream>>>(allocations[0], d_in, d_out,
                                                                                           num_items, num_tiles,
                                                                                           temp_size);


    }
}


#endif //CUBSCAN_DO_SCAN_CUH
