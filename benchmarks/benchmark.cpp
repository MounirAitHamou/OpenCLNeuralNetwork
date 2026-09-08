#include "clnn/clnn.hpp"

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

double run_benchmark(const std::string& name, const clnn::Device device, const int iterations,
                     const std::function<void()>& operation) {
    for (int iteration = 0; iteration < 2; ++iteration)
        operation();
    device.synchronize();
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
        operation();
    device.synchronize();
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start);
    const auto milliseconds = elapsed.count() / static_cast<double>(iterations);
    std::cout << std::left << std::setw(28) << name << std::right << std::fixed
              << std::setprecision(3) << milliseconds << " ms/iteration\n";
    return milliseconds;
}

bool benchmark_device(const clnn::Device device, const bool regression_check) {
    std::cout << "\n" << device.name() << "\n";
    if (device.type() == clnn::DeviceType::opencl) {
        device.set_profiling(true);
        static_cast<void>(device.profile());
    }
    std::mt19937 generator(42);
    const auto lhs = clnn::Tensor::randn({128, 128}, generator, 0.0F, 1.0F, false, {}, device);
    const auto rhs = clnn::Tensor::randn({128, 128}, generator, 0.0F, 1.0F, false, {}, device);
    const auto image = clnn::Tensor::randn({8, 3, 32, 32}, generator, 0.0F, 1.0F, false, {}, device);
    const auto weight = clnn::Tensor::randn({16, 3, 3, 3}, generator, 0.0F, 0.1F, false, {}, device);
    const auto bias = clnn::Tensor::zeros({16}, false, {}, device);
    clnn::NoGradGuard no_grad;

    std::vector<double> timings;
    timings.push_back(run_benchmark("128x128 matrix multiply", device,
                                    regression_check ? 3 : 20, [&] {
        static_cast<void>(clnn::matmul(lhs, rhs));
    }));
    timings.push_back(run_benchmark("conv + relu + max pool", device,
                                    regression_check ? 2 : 20, [&] {
        static_cast<void>(clnn::max_pool2d(clnn::relu(clnn::conv2d(image, weight, bias, {1, 1},
                                                                  {1, 1})),
                                           {2, 2}));
    }));
    timings.push_back(run_benchmark("axis reduction", device, regression_check ? 3 : 50, [&] {
        static_cast<void>(clnn::mean(image, 0, true));
    }));

    if (device.type() == clnn::DeviceType::opencl) {
        const auto profiles = device.profile();
        std::cout << "  OpenCL kernel events\n";
        for (const auto& profile : profiles) {
            std::cout << "    " << std::left << std::setw(22) << profile.operation << std::right
                      << std::setw(5) << profile.calls << " calls, " << std::setw(9)
                      << profile.total_milliseconds << " ms total\n";
        }
        const auto statistics = device.runtime_statistics();
        std::cout << "  runtime: " << statistics.cached_kernels << " cached kernels, "
                  << statistics.buffer_allocations << " allocations, " << statistics.buffer_reuses
                  << " buffer reuses, " << statistics.pooled_buffers << " pooled buffers\n";
        device.set_profiling(false);
    }

    if (!regression_check)
        return true;
    constexpr double maximum_milliseconds[] = {3000.0, 5000.0, 2000.0};
    for (std::size_t index = 0; index < timings.size(); ++index) {
        if (!std::isfinite(timings[index]) || timings[index] > maximum_milliseconds[index]) {
            std::cerr << "benchmark regression: case " << index << " exceeded "
                      << maximum_milliseconds[index] << " ms/iteration\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main(const int argc, const char* const* argv) {
    const bool regression_check = argc > 1 && std::string(argv[1]) == "--check";
    auto success = benchmark_device(clnn::Device::cpu(), regression_check);
    if (clnn::opencl_available())
        success = benchmark_device(clnn::Device::opencl(), regression_check) && success;
    else
        std::cout << "\nOpenCL GPU unavailable; GPU benchmarks skipped.\n";
    return success ? 0 : 1;
}
