// cpu_vs_cuda_3d_benchmark_parallel_cpu.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <thread>

#define CHECK_CUDA(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); }} while(0)

__global__ void kernel3D(const float* in, float* out,
                         int X, int Y, int Z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < X && y < Y && z < Z) {
        int idx = z * (X * Y) + y * X + x;
        float v = in[idx];
        out[idx] = sinf(v) * cosf(v) + sqrtf(v + 1.0f);
    }
}

void cpu_worker(const float* in, float* out,
                int X, int Y, int Z,
                int z_start, int z_end)
{
    for (int z = z_start; z < z_end; ++z)
        for (int y = 0; y < Y; ++y)
            for (int x = 0; x < X; ++x) {
                int idx = z * (X * Y) + y * X + x;
                float v = in[idx];
                out[idx] = std::sin(v) * std::cos(v) + std::sqrt(v + 1.0f);
            }
}

int main()
{
    const int X = 2560;
    const int Y = 2560;
    const int Z = 64;
    const size_t N = static_cast<size_t>(X) * Y * Z;
    const size_t bytes = N * sizeof(float);
    const int iterations = 50;

    std::vector<float> h_in(N), h_out_cpu(N), h_out_gpu(N);

    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i % 100) * 0.01f;

    // ================= PARALLEL CPU =================
    const unsigned num_threads = std::min(16u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        threads.clear();
        int z_chunk = Z / num_threads;

        for (unsigned t = 0; t < num_threads; ++t) {
            int z_start = t * z_chunk;
            int z_end = (t == num_threads - 1) ? Z : z_start + z_chunk;

            threads.emplace_back(cpu_worker,
                h_in.data(), h_out_cpu.data(),
                X, Y, Z, z_start, z_end);
        }

        for (auto& th : threads)
            th.join();
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();

    // ================= CUDA =================
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(8, 8, 8);
    dim3 grid(
        (X + block.x - 1) / block.x,
        (Y + block.y - 1) / block.y,
        (Z + block.z - 1) / block.z
    );

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

    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "CPU threads: " << num_threads << std::endl;
    std::cout << "CPU time (ms):  " << cpu_ms << std::endl;
    std::cout << "CUDA time (ms): " << gpu_ms << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
