//
// Created by yxgao on 2022/3/9.
//

#include "cub_full/cub/cub.cuh"   // or equivalently <cub/device/device_scan.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <sys/time.h>
#include <iostream>
// Declare, allocate, and initialize device-accessible pointers for input and output
#define test_type int
#define test_times 1000
using namespace std;
int main(int argc ,char **argv) {
    int test_num_items = atoi(argv[1]);      // e.g., 7
    int *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
    int *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
    thrust::host_vector<test_type> h_vec(test_num_items);
    thrust::host_vector<test_type> h_vec_flag(test_num_items);

    for (int i = 0; i < test_num_items; ++i) {
        h_vec[i] = 1;
    }
    thrust::device_vector<test_type> d_vec(h_vec);
    thrust::device_vector<test_type> d_vec_flag(h_vec_flag);
    d_in = d_vec.data().get();
    d_out = d_vec_flag.data().get();
// Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub_new_compiled::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, test_num_items);
// Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// Run exclusive prefix sum

    struct timeval t1, t2;
    gettimeofday(&t1, nullptr);
    for (int i = 0; i < test_times; ++i) {
        cub_new_compiled::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, test_num_items);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, nullptr);
    double time = 1000 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << test_num_items * sizeof(test_type) * test_times / (time *1e6) << endl;
    cout << time / test_times << endl;
// d_out s<-- [0, 8, 14, 21, 26, 29, 29]
}