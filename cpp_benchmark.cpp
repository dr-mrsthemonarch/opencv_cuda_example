// cpu_benchmark.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>
#include  <cstdlib>

#include "cmake-build-debug/_deps/opencv-build/3rdparty/ade/ade-0.1.2e/sources/ade/include/ade/memory/alloc.hpp"

// CPU worker for a chunk of the 3D array
void cpu_worker(const float *in, float *out,
                int X, int Y, int Z,
                int z_start, int z_end) {
    for (int z = z_start; z < z_end; ++z)
        for (int y = 0; y < Y; ++y)
            for (int x = 0; x < X; ++x) {
                int idx = z * (X * Y) + y * X + x;
                float v = in[idx];
                out[idx] = std::sin(v) * std::cos(v) + std::sqrt(v + 1.0f);
            }
}


void run_cpu() {
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

        for (auto &th: threads)
            th.join();
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();


    std::cout << "CPU time (ms): " << cpu_ms << std::endl;
}

void run_cpu_fast() {
    const int X = 2560;
    const int Y = 2560;
    const int Z = 64;
    const size_t N = static_cast<size_t>(X) * Y * Z;
    const int iterations = 50;

    std::vector<float> h_in(N), h_out_cpu(N);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i % 100) * 0.01f;

    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iterations; ++it) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            float v = h_in[i];
            h_out_cpu[i] = std::sin(v) * std::cos(v) + std::sqrt(v + 1.0f);
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "CPU time (ms, OpenMP): " << cpu_ms << std::endl;
}

// Even faster version with manual SIMD
void run_cpu_simd() {
    const int X = 2560;
    const int Y = 2560;
    const int Z = 64;
    const size_t N = static_cast<size_t>(X) * Y * Z;
    const int iterations = 50;

    // Aligned allocation for SIMD
    float *h_in = (float *) _mm_malloc(N * sizeof(float), 64);
    float *h_out_cpu = (float *) _mm_malloc(N * sizeof(float), 64);

    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i % 100) * 0.01f;

    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iterations; ++it) {
#pragma omp parallel
        {
#pragma omp for schedule(static) nowait
            for (int i = 0; i < N; i += 8) {
                if (i + 8 <= N) {
                    // Load 8 floats
                    __m256 v = _mm256_load_ps(&h_in[i]);

                    // Compute sin and cos (using SVML if available, otherwise scalar fallback)
                    __m256 sin_v = _mm256_sin_ps(v);
                    __m256 cos_v = _mm256_cos_ps(v);

                    // v + 1.0f
                    __m256 v_plus_one = _mm256_add_ps(v, _mm256_set1_ps(1.0f));

                    // sqrt(v + 1.0f)
                    __m256 sqrt_v = _mm256_sqrt_ps(v_plus_one);

                    // sin * cos + sqrt
                    __m256 result = _mm256_fmadd_ps(sin_v, cos_v, sqrt_v);

                    // Store result
                    _mm256_store_ps(&h_out_cpu[i], result);
                } else {
                    // Handle remainder
                    for (size_t j = i; j < N; ++j) {
                        float v = h_in[j];
                        h_out_cpu[j] = std::sin(v) * std::cos(v) + std::sqrt(v + 1.0f);
                    }
                }
            }
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();


    std::cout << "CPU time (ms, SIMD): " << cpu_ms << std::endl;

    _mm_free(h_in);
    _mm_free(h_out_cpu);
}

// Helper for aligned allocator (add to top of file)
template<typename T, size_t Alignment>
struct aligned_allocator {
    using value_type = T;

    T *allocate(size_t n) {
        return static_cast<T *>(_mm_malloc(n * sizeof(T), Alignment));
    }

    void deallocate(T *p, size_t) {
        _mm_free(p);
    }
};


// Optimized for AMD: better cache blocking + simplified math
void run_cpu_amd_optimized() {
    const int X = 2560;
    const int Y = 2560;
    const int Z = 64;
    const size_t N = static_cast<size_t>(X) * Y * Z;
    const int iterations = 50;

    // Aligned allocation
    float *h_in = (float *) _mm_malloc(N * sizeof(float), 64);
    float *h_out_cpu = (float *) _mm_malloc(N * sizeof(float), 64);

    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i % 100) * 0.01f;

    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Move parallel region OUTSIDE iteration loop (critical for AMD)
#pragma omp parallel num_threads(16)
    {
        for (int it = 0; it < iterations; ++it) {
            // Static scheduling, large chunks for better cache locality
#pragma omp for schedule(static, 4096) nowait
            for (int i = 0; i < N; i += 8) {
                if (i + 8 <= N) {
                    __m256 v = _mm256_loadu_ps(&h_in[i]);

                    // Simplified: sin(v)*cos(v) = 0.5*sin(2v)
                    __m256 two_v = _mm256_add_ps(v, v);

                    // For AMD without SVML, use fast approximations
                    // or let compiler handle it with -ffast-math
                    __m256 sin_2v = _mm256_sin_ps(two_v);
                    __m256 half_sin = _mm256_mul_ps(sin_2v, _mm256_set1_ps(0.5f));

                    __m256 v_plus_one = _mm256_add_ps(v, _mm256_set1_ps(1.0f));
                    __m256 sqrt_v = _mm256_sqrt_ps(v_plus_one);

                    __m256 result = _mm256_add_ps(half_sin, sqrt_v);

                    _mm256_storeu_ps(&h_out_cpu[i], result);
                } else {
                    for (size_t j = i; j < N; ++j) {
                        float v = h_in[j];
                        h_out_cpu[j] = 0.5f * std::sin(2.0f * v) + std::sqrt(v + 1.0f);
                    }
                }
            }
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();


    std::cout << "CPU time (ms, AMD optmized): " << cpu_ms << std::endl;

    _mm_free(h_in);
    _mm_free(h_out_cpu);
}

// Fast approximation version (if SVML not available)
inline __m256 sin_ps_approx(__m256 x) {
    // Bhaskara I's sine approximation (good for small angles)
    // For full range, need range reduction first
    const __m256 pi = _mm256_set1_ps(3.14159265f);
    const __m256 two_pi = _mm256_set1_ps(6.28318531f);

    // Range reduction: wrap to [-pi, pi]
    __m256 k = _mm256_round_ps(_mm256_div_ps(x, two_pi),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_sub_ps(x, _mm256_mul_ps(k, two_pi));

    // Taylor series: sin(x) ≈ x - x³/6 + x⁵/120
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 x5 = _mm256_mul_ps(x3, x2);

    __m256 result = x;
    result = _mm256_sub_ps(result, _mm256_mul_ps(x3, _mm256_set1_ps(0.166666667f)));
    result = _mm256_add_ps(result, _mm256_mul_ps(x5, _mm256_set1_ps(0.008333333f)));

    return result;
}

void run_cpu_amd_fast_math() {
    const int X = 2560;
    const int Y = 2560;
    const int Z = 64;
    const size_t N = static_cast<size_t>(X) * Y * Z;
    const int iterations = 50;

    float *h_in = (float *) _mm_malloc(N * sizeof(float), 64);
    float *h_out_cpu = (float *) _mm_malloc(N * sizeof(float), 64);

    for (int i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i % 100) * 0.01f;

    auto cpu_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(16)
    {
        for (int it = 0; it < iterations; ++it) {
#pragma omp for schedule(static, 4096) nowait
            for (int i = 0; i < N; i += 8) {
                if (i + 8 <= N) {
                    __m256 v = _mm256_loadu_ps(&h_in[i]);
                    __m256 two_v = _mm256_add_ps(v, v);

                    // Use fast approximation
                    __m256 half_sin = _mm256_mul_ps(sin_ps_approx(two_v),
                                                    _mm256_set1_ps(0.5f));

                    __m256 v_plus_one = _mm256_add_ps(v, _mm256_set1_ps(1.0f));
                    __m256 sqrt_v = _mm256_sqrt_ps(v_plus_one);

                    __m256 result = _mm256_add_ps(half_sin, sqrt_v);
                    _mm256_storeu_ps(&h_out_cpu[i], result);
                } else {
                    for (int j = i; j < N; ++j) {
                        float v = h_in[j];
                        h_out_cpu[j] = 0.5f * std::sin(2.0f * v) + std::sqrt(v + 1.0f);
                    }
                }
            }
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "CPU time (ms, AMD Fast Math): " << cpu_ms << std::endl;

    _mm_free(h_in);
    _mm_free(h_out_cpu);
}

void run_cpu_amd_simple2() {
    const int X = 2560;
    const int Y = 2560;
    const int Z = 64;
    const size_t N = static_cast<size_t>(X) * Y * Z;
    const int iterations = 50;

    float*  h_in = (float*)_mm_malloc(N * sizeof(float), 64);
    float*  h_out_cpu = (float*)_mm_malloc(N * sizeof(float), 64);

    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i % 100) * 0.01f;

    auto cpu_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(16)
    {
        for (int it = 0; it < iterations; ++it) {
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < N; ++i) {
                float v = h_in[i];
                // Let compiler vectorize with -ffast-math -O3
                h_out_cpu[i] = 0.5f * std::sin(2.0f * v) + std::sqrt(v + 1.0f);
            }
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "CPU time (ms, AMD Compiler Optimized): " << cpu_ms << std::endl;

    _mm_free(h_in);
    _mm_free(h_out_cpu);
}

void run_cpu_amd_optimized2() {
    const int X = 2560;
    const int Y = 2560;
    const int Z = 64;
    const size_t N = static_cast<size_t>(X) * Y * Z;
    const int iterations = 50;

    // Aligned allocation
    float* h_in = (float*)_mm_malloc(N * sizeof(float), 64);
    float* h_out_cpu = (float*)_mm_malloc(N * sizeof(float), 64);

    for (size_t i = 0; i < N; ++i)
        h_in[i] = static_cast<float>(i % 100) * 0.01f;

    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Move parallel region OUTSIDE iteration loop (critical for AMD)
    #pragma omp parallel num_threads(16)
    {
        for (int it = 0; it < iterations; ++it) {
            // Static scheduling, large chunks for better cache locality
            #pragma omp for schedule(static, 4096) nowait
            for (size_t i = 0; i < N; i += 8) {
                if (i + 8 <= N) {
                    __m256 v = _mm256_loadu_ps(&h_in[i]);

                    // Simplified: sin(v)*cos(v) = 0.5*sin(2v)
                    __m256 two_v = _mm256_add_ps(v, v);

                    // For AMD without SVML, use fast approximations
                    // or let compiler handle it with -ffast-math
                    __m256 sin_2v = _mm256_sin_ps(two_v);
                    __m256 half_sin = _mm256_mul_ps(sin_2v, _mm256_set1_ps(0.5f));

                    __m256 v_plus_one = _mm256_add_ps(v, _mm256_set1_ps(1.0f));
                    __m256 sqrt_v = _mm256_sqrt_ps(v_plus_one);

                    __m256 result = _mm256_add_ps(half_sin, sqrt_v);

                    _mm256_storeu_ps(&h_out_cpu[i], result);
                } else {
                    for (size_t j = i; j < N; ++j) {
                        float v = h_in[j];
                        h_out_cpu[j] = 0.5f * std::sin(2.0f * v) + std::sqrt(v + 1.0f);
                    }
                }
            }
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "CPU time (ms, AMD Optimized 2): " << cpu_ms << std::endl;

    _mm_free(h_in);
    _mm_free(h_out_cpu);
}
