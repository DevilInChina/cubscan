#include <iostream>
#include "do_scan.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <ctime>
#include <sys/time.h>
#include <iostream>


using namespace std;
#define test_type int
int main(int argc,char **argv) {
    int test_num_items = atoi(argv[1]);      // e.g., 7
    thrust::host_vector<test_type> h_vec(test_num_items);
    thrust::host_vector<test_type> h_vec_flag(test_num_items);

    for (int i = 0; i < test_num_items; ++i) {
        h_vec[i] = 1;
    }
    thrust::device_vector<test_type> d_vec(h_vec);
    thrust::device_vector<test_type> d_vec_flag(h_vec_flag);


    thrust::device_vector<test_type> d_vec_res(test_num_items);
    cudaStream_t cudaStream;
    cudaStreamCreate(&cudaStream);
    void *temp = nullptr;
    unsigned int temp_size = scan::get_temp_storage_size<test_type>(test_num_items);
    cudaMalloc(&temp, temp_size);
    cudaDeviceSynchronize();
    struct timeval t1, t2;
    gettimeofday(&t1, nullptr);
    for (int i = 0; i < 10; ++i) {
        scan::ExclusiveScan(temp, d_vec.data().get(), d_vec_res.data().get(), test_num_items, cudaStream);
    }
    cudaStreamSynchronize(cudaStream);
    gettimeofday(&t2, nullptr);
    thrust::host_vector<test_type> h_vec_res(test_num_items);
    double time = 1000 * (t2.tv_sec - t2.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << test_num_items * sizeof(test_type) * 10 / (time *1e6) << endl;
    cudaDeviceSynchronize();
    h_vec_flag = d_vec_res;
    cudaDeviceSynchronize();

    return 0;
}
