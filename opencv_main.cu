// opencv_cpu_vs_cuda_edges_morphology.cpp
#define OPENCV_DISABLE_LOGGING

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#include "_deps/opencv_contrib-src/modules/cudafilters/include/opencv2/cudafilters.hpp"
#include "_deps/opencv_contrib-src/modules/cudaimgproc/include/opencv2/cudaimgproc.hpp"

int main(int argc, char** argv){
    cv::utils::logging::setLogLevel(
    cv::utils::logging::LOG_LEVEL_SILENT);


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

    // ================= CPU PIPELINE =================
    cv::Mat cpu_blur, cpu_edges, cpu_morph;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cv::GaussianBlur(image, cpu_blur, cv::Size(7, 7), 0);
        cv::Canny(cpu_blur, cpu_edges, 50, 150);
        cv::morphologyEx(cpu_edges, cpu_morph, cv::MORPH_CLOSE, kernel);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();

    // ================= CUDA PIPELINE =================
    cv::cuda::GpuMat d_img, d_blur, d_edges, d_morph;
    d_img.upload(image);

    auto gaussian = cv::cuda::createGaussianFilter(
        d_img.type(), d_img.type(), cv::Size(7, 7), 0);

    auto canny = cv::cuda::createCannyEdgeDetector(50.0, 150.0);

    auto morph = cv::cuda::createMorphologyFilter(
        cv::MORPH_CLOSE, d_img.type(), kernel);

    cv::cuda::Event start, stop;
    start.record();
    for (int i = 0; i < iterations; ++i) {
        gaussian->apply(d_img, d_blur);
        canny->detect(d_blur, d_edges);
        morph->apply(d_edges, d_morph);
    }
    stop.record();
    cudaDeviceSynchronize();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    float gpu_ms = cv::cuda::Event::elapsedTime(start, stop);

    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "CPU time (ms):  " << cpu_ms << std::endl;
    std::cout << "CUDA time (ms): " << gpu_ms << std::endl;

    cv::Mat result;
    d_morph.download(result);

    cv::imshow("CUDA Result", result);
    cv::waitKey(0);

    return 0;
}
