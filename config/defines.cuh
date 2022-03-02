#include "cub_origin/cub.cuh"

#ifndef CUBSCAN_DEFINES_CUH
namespace scan {
    template<int A>
    using Int2Type = origin_cub::Int2Type<A>;

    template<typename T>
    using Traits = origin_cub::Traits<T>;

    template<typename T>
    using Uninitialized = origin_cub::Uninitialized<T>;

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

    template<
            typename _Key,
            typename _Value>
    using KeyValuePair = origin_cub::KeyValuePair<_Key, _Value>;

}
#define CUBSCAN_DEFINES_CUH
#endif