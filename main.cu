#include <iostream>
#include "block_scan.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

__global__ void test(const int *data, const int *flag, int *res) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int ss;
    scan::BlockScanWarpScans<int,128,1,1,CUB_PTX_ARCH>().InclusiveScan(data[id], res[id],ss);
}

using namespace std;

int main() {
    thrust::host_vector<int> h_vec(256);
    thrust::host_vector<int> h_vec_flag(256);

    for (int i = 0; i < 256; ++i) {
        h_vec[i] = 1;
    }
    thrust::device_vector<int> d_vec(h_vec);
    thrust::device_vector<int> d_vec_flag(h_vec_flag);


    thrust::device_vector<int> d_vec_res(256);
    test<<<2, 128>>>(d_vec.data().get(), d_vec_flag.data().get(), d_vec_res.data().get());
    thrust::host_vector<int> h_vec_res(256);
    cudaDeviceSynchronize();
    h_vec_flag = d_vec_res;
    cudaDeviceSynchronize();
    for (int i = 0; i < 256; ++i) {
        cout << i << "\t" << h_vec[i] << "\t" << d_vec_res[i] << "\n";
    }

    return 0;
}
