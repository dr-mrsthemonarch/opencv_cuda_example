// main.cpp
#include <string>

// Forward declarations
void run_cpu();

void run_cpu_fast();

void run_gpu();

void run_cpu_simd();

void run_cpu_amd_optimized();

void run_cpu_amd_fast_math();

void run_cpu_amd_compiler();

void run_cpu_amd_optimized2();

int main(int argc, char **argv) {
    run_gpu();
    run_cpu();
    run_cpu_fast();
    run_cpu_simd();
    run_cpu_amd_optimized();
    run_cpu_amd_optimized2();
    run_cpu_amd_fast_math();
    run_cpu_amd_compiler();
    return 0;
}

