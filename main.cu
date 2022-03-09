#include <iostream>
#include "do_scan.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>


using namespace std;
#define test_num_items (128*1024)
int main() {
    thrust::host_vector<int> h_vec(test_num_items);
    thrust::host_vector<int> h_vec_flag(test_num_items);

    for (int i = 0; i < test_num_items; ++i) {
        h_vec[i] = 1;
    }
    thrust::device_vector<int> d_vec(h_vec);
    thrust::device_vector<int> d_vec_flag(h_vec_flag);


    thrust::device_vector<int> d_vec_res(test_num_items);
    cudaStream_t cudaStream;
    cudaStreamCreate(&cudaStream);
    void *temp = nullptr;
    unsigned int temp_size = scan::get_temp_storage_size<int>(test_num_items);
    cudaMalloc(&temp,temp_size);
    cudaDeviceSynchronize();
    scan::ExclusiveScan(temp,d_vec.data().get(), d_vec_res.data().get(), test_num_items,cudaStream);
    thrust::host_vector<int> h_vec_res(test_num_items);

    cudaDeviceSynchronize();
    h_vec_flag = d_vec_res;
    cudaDeviceSynchronize();
    int cnt = 0;
    for (int i = 0; i < test_num_items; ++i) {
        if(d_vec_res[i] != i && ++cnt < 20)
            cout << i << "\t" << h_vec[i] << "\t" << d_vec_res[i] << "\n";
    }

    return 0;
}
