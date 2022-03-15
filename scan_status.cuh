//
// Created by yxgao on 2022/3/9.
//
#include "defines.cuh"
#ifndef CUBSCAN_SCAN_STATUS_CUH
#define CUBSCAN_SCAN_STATUS_CUH
namespace scan{
    template <
            typename    T,
            bool        SINGLE_WORD = Traits<T>::PRIMITIVE>
    struct ScanTileState;

    template <typename T>
    struct ScanTileState<T, true>
    {
        // Status word type
        using StatusWord = std::conditional_t<
                sizeof(T) == 8,
                long long,
                std::conditional_t<
                        sizeof(T) == 4,
                        int,
                        std::conditional_t<sizeof(T) == 2, short, char>>>;

        // Unit word type
        using TxnWord = std::conditional_t<
                sizeof(T) == 8,
                longlong2,
                std::conditional_t<
                        sizeof(T) == 4,
                        int2,
                        std::conditional_t<sizeof(T) == 2, int, uchar2>>>;

        // Device word type
        struct TileDescriptor
        {
            StatusWord  status;
            T           value;
        };


        // Constants
        enum
        {
            TILE_STATUS_PADDING = CUB_PTX_WARP_THREADS,
        };


        // Device storage
        TxnWord *d_tile_descriptors;

        /// Constructor
        __host__ __device__ __forceinline__
        ScanTileState()
                :
                d_tile_descriptors(NULL)
        {}


        /// Initializer
        __host__ __device__ __forceinline__
        cudaError_t Init(
                int     /*num_tiles*/,                      ///< [in] Number of tiles
                void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                size_t  /*temp_storage_bytes*/)             ///< [in] Size in bytes of \t d_temp_storage allocation
        {
            d_tile_descriptors = reinterpret_cast<TxnWord*>(d_temp_storage);
            return cudaSuccess;
        }


        /**
         * Compute device memory needed for tile status
         */
        __host__ __device__ __forceinline__
        static cudaError_t AllocationSize(
                int     num_tiles,                          ///< [in] Number of tiles
                size_t  &temp_storage_bytes)                ///< [out] Size in bytes of \t d_temp_storage allocation
        {
            temp_storage_bytes = (num_tiles + TILE_STATUS_PADDING) * sizeof(TileDescriptor);       // bytes needed for tile status descriptors
            return cudaSuccess;
        }


        /**
         * Initialize (from device)
         */
        __device__ __forceinline__ void InitializeStatus(int num_tiles)
        {
            int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

            TxnWord val = TxnWord();
            TileDescriptor *descriptor = reinterpret_cast<TileDescriptor*>(&val);

            if (tile_idx < num_tiles)
            {
                // Not-yet-set
                descriptor->status = StatusWord(SCAN_TILE_INVALID);
                d_tile_descriptors[TILE_STATUS_PADDING + tile_idx] = val;
            }

            if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
            {
                // Padding
                descriptor->status = StatusWord(SCAN_TILE_OOB);
                d_tile_descriptors[threadIdx.x] = val;
            }
        }


        /**
         * Update the specified tile's inclusive value and corresponding status
         */
        __device__ __forceinline__ void SetInclusive(int tile_idx, T tile_inclusive)
        {
            TileDescriptor tile_descriptor;
            tile_descriptor.status = SCAN_TILE_INCLUSIVE;
            tile_descriptor.value = tile_inclusive;

            TxnWord alias;
            *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;
            CUB_NS_QUALIFIER::ThreadStore<CUB_NS_QUALIFIER::STORE_CG>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, alias);
        }


        /**
         * Update the specified tile's partial value and corresponding status
         */
        __device__ __forceinline__ void SetPartial(int tile_idx, T tile_partial)
        {
            TileDescriptor tile_descriptor;
            tile_descriptor.status = SCAN_TILE_PARTIAL;
            tile_descriptor.value = tile_partial;

            TxnWord alias;
            *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;
            CUB_NS_QUALIFIER::ThreadStore<CUB_NS_QUALIFIER::STORE_CG>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, alias);
        }

        /**
         * Wait for the corresponding tile to become non-invalid
         */
        __device__ __forceinline__ void WaitForValid(
                int             tile_idx,
                StatusWord      &status,
                T               &value)
        {
            TileDescriptor tile_descriptor;
            do
            {
                __threadfence_block(); // prevent hoisting loads from loop
                TxnWord alias = CUB_NS_QUALIFIER::ThreadLoad<CUB_NS_QUALIFIER::LOAD_CG>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
                tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);

            } while (CUB_NS_QUALIFIER::WARP_ANY((tile_descriptor.status == SCAN_TILE_INVALID), 0xffffffff));

            status = tile_descriptor.status;
            value = tile_descriptor.value;
        }

    };
    template <typename T>
    struct ScanTileState<T, false>
    {
        // Status word type
        typedef char StatusWord;

        // Constants
        enum
        {
            TILE_STATUS_PADDING = CUB_PTX_WARP_THREADS,
        };

        // Device storage
        StatusWord  *d_tile_status;
        T           *d_tile_partial;
        T           *d_tile_inclusive;

        /// Constructor
        __host__ __device__ __forceinline__
        ScanTileState()
                :
                d_tile_status(NULL),
                d_tile_partial(NULL),
                d_tile_inclusive(NULL)
        {}


        /// Initializer
        __host__ __device__ __forceinline__
        cudaError_t Init(
                int     num_tiles,                          ///< [in] Number of tiles
                void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
                size_t  temp_storage_bytes)                 ///< [in] Size in bytes of \t d_temp_storage allocation
        {
            cudaError_t error = cudaSuccess;
            do
            {
                void*   allocations[3] = {};
                size_t  allocation_sizes[3];

                allocation_sizes[0] = (num_tiles + TILE_STATUS_PADDING) * sizeof(StatusWord);           // bytes needed for tile status descriptors
                allocation_sizes[1] = (num_tiles + TILE_STATUS_PADDING) * sizeof(CUB_NS_QUALIFIER::Uninitialized<T>);     // bytes needed for partials
                allocation_sizes[2] = (num_tiles + TILE_STATUS_PADDING) * sizeof(CUB_NS_QUALIFIER::Uninitialized<T>);     // bytes needed for inclusives

                // Compute allocation pointers into the single storage blob
                if (CubDebug(error = CUB_NS_QUALIFIER::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

                // Alias the offsets
                d_tile_status       = reinterpret_cast<StatusWord*>(allocations[0]);
                d_tile_partial      = reinterpret_cast<T*>(allocations[1]);
                d_tile_inclusive    = reinterpret_cast<T*>(allocations[2]);
            }
            while (0);

            return error;
        }


        /**
         * Compute device memory needed for tile status
         */
        __host__ __device__ __forceinline__
        static cudaError_t AllocationSize(
                int     num_tiles,                          ///< [in] Number of tiles
                size_t  &temp_storage_bytes)                ///< [out] Size in bytes of \t d_temp_storage allocation
        {
            // Specify storage allocation requirements
            size_t  allocation_sizes[3];
            allocation_sizes[0] = (num_tiles + TILE_STATUS_PADDING) * sizeof(StatusWord);         // bytes needed for tile status descriptors
            allocation_sizes[1] = (num_tiles + TILE_STATUS_PADDING) * sizeof(CUB_NS_QUALIFIER::Uninitialized<T>);   // bytes needed for partials
            allocation_sizes[2] = (num_tiles + TILE_STATUS_PADDING) * sizeof(CUB_NS_QUALIFIER::Uninitialized<T>);   // bytes needed for inclusives

            // Set the necessary size of the blob
            void* allocations[3] = {};
            return CubDebug(CUB_NS_QUALIFIER::AliasTemporaries(NULL, temp_storage_bytes, allocations, allocation_sizes));
        }


        /**
         * Initialize (from device)
         */
        __device__ __forceinline__ void InitializeStatus(int num_tiles)
        {
            int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (tile_idx < num_tiles)
            {
                // Not-yet-set
                d_tile_status[TILE_STATUS_PADDING + tile_idx] = StatusWord(SCAN_TILE_INVALID);
            }

            if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
            {
                // Padding
                d_tile_status[threadIdx.x] = StatusWord(SCAN_TILE_OOB);
            }
        }


        /**
         * Update the specified tile's inclusive value and corresponding status
         */
        __device__ __forceinline__ void SetInclusive(int tile_idx, T tile_inclusive)
        {
            // Update tile inclusive value
            CUB_NS_QUALIFIER::ThreadStore<CUB_NS_QUALIFIER::STORE_CG>(d_tile_inclusive + TILE_STATUS_PADDING + tile_idx, tile_inclusive);

            // Fence
            __threadfence();

            // Update tile status
            CUB_NS_QUALIFIER::ThreadStore<CUB_NS_QUALIFIER::STORE_CG>(d_tile_status + TILE_STATUS_PADDING + tile_idx, StatusWord(SCAN_TILE_INCLUSIVE));
        }


        /**
         * Update the specified tile's partial value and corresponding status
         */
        __device__ __forceinline__ void SetPartial(int tile_idx, T tile_partial)
        {
            // Update tile partial value
            CUB_NS_QUALIFIER::ThreadStore<CUB_NS_QUALIFIER::STORE_CG>(d_tile_partial + TILE_STATUS_PADDING + tile_idx, tile_partial);

            // Fence
            __threadfence();

            // Update tile status
            CUB_NS_QUALIFIER::ThreadStore<CUB_NS_QUALIFIER::STORE_CG>(d_tile_status + TILE_STATUS_PADDING + tile_idx, StatusWord(SCAN_TILE_PARTIAL));
        }

        /**
         * Wait for the corresponding tile to become non-invalid
         */
        __device__ __forceinline__ void WaitForValid(
                int             tile_idx,
                StatusWord      &status,
                T               &value)
        {
            do {
                status = CUB_NS_QUALIFIER::ThreadLoad<CUB_NS_QUALIFIER::LOAD_CG>(d_tile_status + TILE_STATUS_PADDING + tile_idx);

                __threadfence();    // prevent hoisting loads from loop or loads below above this one

            } while (status == SCAN_TILE_INVALID);

            if (status == StatusWord(SCAN_TILE_PARTIAL))
                value = CUB_NS_QUALIFIER::ThreadLoad<CUB_NS_QUALIFIER::LOAD_CG>(d_tile_partial + TILE_STATUS_PADDING + tile_idx);
            else
                value = CUB_NS_QUALIFIER::ThreadLoad<CUB_NS_QUALIFIER::LOAD_CG>(d_tile_inclusive + TILE_STATUS_PADDING + tile_idx);
        }
    };

}
#endif //CUBSCAN_SCAN_STATUS_CUH
