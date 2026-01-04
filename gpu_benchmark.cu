// gpu_benchmark.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); }} while(0)

__global__ void kernel3D(const float* in, float* out, int X, int Y, int Z) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < X && y < Y && z < Z) {
        int idx = z * (X * Y) + y * X + x;
        float v = in[idx];
        out[idx] = sinf(v) * cosf(v) + sqrtf(v + 1.0f);
    }
}

void run_gpu() {
    const int X = 2560, Y = 2560, Z = 64;
    const size_t N = static_cast<size_t>(X) * Y * Z;
    const size_t bytes = N * sizeof(float);
    const int iterations = 50;

    std::vector<float> h_in(N), h_out_gpu(N);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i % 100) * 0.01f;

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32, 32);
    dim3 grid((X + block.x - 1) / block.x,
              (Y + block.y - 1) / block.y,
              (Z + block.z - 1) / block.z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int it = 0; it < iterations; ++it) {
        kernel3D<<<grid, block>>>(d_in, d_out, X, Y, Z);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    std::cout << "CUDA time (ms): " << gpu_ms << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
