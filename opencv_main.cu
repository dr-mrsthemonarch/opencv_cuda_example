// opencv_cpu_vs_cuda_edges_morphology_parallel.cpp
#include <opencv2/opencv.hpp>

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#include "_deps/opencv_contrib-src/modules/cudafilters/include/opencv2/cudafilters.hpp"
#include "_deps/opencv_contrib-src/modules/cudaimgproc/include/opencv2/cudaimgproc.hpp"

#define CHECK_CUDA(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); }} while(0)

// CPU worker for a slice of the image
void cpu_worker(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel,
                int y_start, int y_end)
{
    cv::Mat slice_in = input.rowRange(y_start, y_end);
    cv::Mat slice_blur, slice_edges, slice_morph;

    cv::GaussianBlur(slice_in, slice_blur, cv::Size(7,7), 0);
    cv::Canny(slice_blur, slice_edges, 50, 150);
    cv::morphologyEx(slice_edges, slice_morph, cv::MORPH_CLOSE, kernel);

    slice_morph.copyTo(output.rowRange(y_start, y_end));
}

int main(int argc, char** argv){
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
        std::cerr << "No CUDA device found\n";
        return -1;
    }

    cv::Mat image = cv::imread(argc > 1 ? argv[1] : "test.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image\n";
        return -1;
    }

    const int iterations = 100;
    const int num_threads = 16; // parallelize across 8 CPU cores

    // ================= CPU PIPELINE (PARALLEL) =================
    cv::Mat cpu_result = cv::Mat::zeros(image.size(), image.type());
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        std::vector<std::thread> threads;
        int rows_per_thread = image.rows / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int y_start = t * rows_per_thread;
            int y_end = (t == num_threads - 1) ? image.rows : y_start + rows_per_thread;
            threads.emplace_back(cpu_worker, std::cref(image), std::ref(cpu_result), std::cref(kernel), y_start, y_end);
        }
        for (auto& th : threads)
            th.join();
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ================= CUDA PIPELINE =================
    cv::cuda::GpuMat d_img, d_blur, d_edges, d_morph;
    d_img.upload(image);

    auto gaussian = cv::cuda::createGaussianFilter(d_img.type(), d_img.type(), cv::Size(7,7), 0);
    auto canny = cv::cuda::createCannyEdgeDetector(50.0, 150.0);
    auto morph = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, d_img.type(), kernel);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gaussian->apply(d_img, d_blur);
        canny->detect(d_blur, d_edges);
        morph->apply(d_edges, d_morph);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));

    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "CPU time (ms) [8 threads]:  " << cpu_ms << std::endl;
    std::cout << "CUDA time (ms):             " << gpu_ms << std::endl;

    // Optional: show result
    cv::Mat result;
    d_morph.download(result);
    cv::imshow("CUDA Result", result);
    cv::waitKey(0);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
