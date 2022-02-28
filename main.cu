#include <iostream>
#include "block_scan.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

__global__ void test(const int *data, const int *flag, int *res) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    scan::warp_scan_shfl<int>().InclusiveScan(data[id], res[id]);
}

using namespace std;

int main() {
    thrust::host_vector<int> h_vec(128);
    thrust::host_vector<int> h_vec_flag(128);

    for (int i = 0; i < 128; ++i) {
        h_vec[i] = i % 32;
    }
    thrust::device_vector<int> d_vec(h_vec);
    thrust::device_vector<int> d_vec_flag(h_vec_flag);


    thrust::device_vector<int> d_vec_res(128);
    test<<<1, 128>>>(d_vec.data().get(), d_vec_flag.data().get(), d_vec_res.data().get());
    thrust::host_vector<int> h_vec_res(128);
    cudaDeviceSynchronize();
    h_vec_flag = d_vec_res;
    cudaDeviceSynchronize();
    for (int i = 0; i < 128; ++i) {
        cout << i << "\t" << h_vec[i] << "\t" << d_vec_res[i] << "\n";
    }

    return 0;
}
