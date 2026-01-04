// gpu_benchmark.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
    exit(1); }} while(0)

// Optimized kernel with better memory coalescing
__global__ void kernel3D(const float* in, float* out, int X, int Y, int Z) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < X && y < Y && z < Z) {
        int idx = z * (X * Y) + y * X + x;
        float v = in[idx];

        // Use standard fast math functions
        out[idx] = sinf(v) * cosf(v) + sqrtf(v + 1.0f);
    }
}

// CPU reference implementation for verification
void cpu_reference(const std::vector<float>& in, std::vector<float>& out, int X, int Y, int Z) {
    size_t N = static_cast<size_t>(X) * Y * Z;
    for (size_t i = 0; i < N; ++i) {
        float v = in[i];
        out[i] = sinf(v) * cosf(v) + sqrtf(v + 1.0f);
    }
}

// Verify results with tolerance
bool verify_results(const std::vector<float>& cpu, const std::vector<float>& gpu, float tolerance = 1e-5f) {
    if (cpu.size() != gpu.size()) return false;

    int errors = 0;
    float max_error = 0.0f;

    for (size_t i = 0; i < cpu.size(); ++i) {
        float diff = fabsf(cpu[i] - gpu[i]);
        max_error = fmaxf(max_error, diff);

        if (diff > tolerance) {
            if (errors < 10) {  // Print first 10 errors
                std::cout << "Mismatch at " << i << ": CPU=" << cpu[i]
                         << " GPU=" << gpu[i] << " diff=" << diff << std::endl;
            }
            errors++;
        }
    }

    std::cout << "Verification: " << (errors == 0 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Total mismatches: " << errors << " / " << cpu.size() << std::endl;

    return errors == 0;
}

void run_gpu() {
    const int X = 2560, Y = 2560, Z = 64;
    const size_t N = static_cast<size_t>(X) * Y * Z;
    const size_t bytes = N * sizeof(float);
    const int iterations = 50;

    // Initialize host data
    std::vector<float> h_in(N), h_out_gpu(N), h_out_cpu(N);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i % 100) * 0.01f;

    // Allocate device memory
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    // Zero out output buffer to detect if kernel runs
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // Verify input was copied correctly
    std::vector<float> verify_input(10);
    CHECK_CUDA(cudaMemcpy(verify_input.data(), d_in, 10 * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Verifying input copy: first value = " << verify_input[0] << " (should be 0)" << std::endl;

    // FIXED: Use proper block dimensions (max 1024 threads per block)
    dim3 block(8, 8, 8);  // 8*8*16 = 1024 threads
    dim3 grid((X + block.x - 1) / block.x,
              (Y + block.y - 1) / block.y,
              (Z + block.z - 1) / block.z);

    std::cout << "Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    std::cout << "Block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << "Total elements: " << N << std::endl;

    // Warmup run
    kernel3D<<<grid, block>>>(d_in, d_out, X, Y, Z);
    // CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int it = 0; it < iterations; ++it) {
        kernel3D<<<grid, block>>>(d_in, d_out, X, Y, Z);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Total CUDA time: " << gpu_ms << " ms for " << iterations << " iterations" << std::endl;
    std::cout << "Average per iteration: " << gpu_ms / iterations << " ms" << std::endl;
    std::cout << "Throughput: " << (N * iterations * sizeof(float) * 2) / (gpu_ms * 1e6) << " GB/s" << std::endl;

    // Verify correctness (only compute subset for speed)
    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "Computing CPU reference (this will take a moment)..." << std::endl;
    cpu_reference(h_in, h_out_cpu, X, Y, Z);
    verify_results(h_out_cpu, h_out_gpu);

    // Sample output values
    std::cout << "\n=== Sample outputs ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        float expected = sinf(h_in[i]) * cosf(h_in[i]) + sqrtf(h_in[i] + 1.0f);
        std::cout << "in[" << i << "]=" << h_in[i]
                  << " -> GPU out=" << h_out_gpu[i]
                  << " | Expected=" << expected
                  << " | Match=" << (fabsf(h_out_gpu[i] - expected) < 1e-4f ? "YES" : "NO")
                  << std::endl;
    }

    // Check if output is all zeros (indicates kernel didn't run)
    bool all_zeros = true;
    for (int i = 0; i < 100 && all_zeros; ++i) {
        if (h_out_gpu[i] != 0.0f) all_zeros = false;
    }
    if (all_zeros) {
        std::cout << "\nWARNING: GPU output appears to be all zeros! Kernel may not have executed properly." << std::endl;
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
