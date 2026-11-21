// Prints the CUDA driver version this process loads and runs one kernel.
// Build:  nvcc -arch=sm_121 -o driver_version driver_version.cu -lcuda
// Check:  ./driver_version                                        -> driver=13000 (stock R580)
//         LD_LIBRARY_PATH=<lab>/compat-root/usr/local/cuda-13.5/compat ./driver_version  -> driver=13050
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fill(int *p) { p[threadIdx.x] = threadIdx.x; }

int main() {
    int driver = 0, runtime = 0;
    cuInit(0);
    cuDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    int *p = nullptr;
    cudaError_t err = cudaMalloc(&p, 128 * sizeof(int));
    if (err == cudaSuccess) {
        fill<<<1, 128>>>(p);
        err = cudaDeviceSynchronize();
    }
    printf("driver=%d runtime=%d kernel=%s\n", driver, runtime, cudaGetErrorString(err));
    return err == cudaSuccess ? 0 : 1;
}
