#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace clnn {

enum class DeviceType { cpu, opencl };

struct KernelProfile final {
    std::string operation;
    std::size_t calls = 0;
    double total_milliseconds = 0.0;
    double minimum_milliseconds = 0.0;
    double maximum_milliseconds = 0.0;
};

struct OpenCLRuntimeStatistics final {
    std::size_t buffer_allocations = 0;
    std::size_t buffer_reuses = 0;
    std::size_t pooled_buffers = 0;
    std::size_t cached_kernels = 0;
};

class Device final {
  public:
    [[nodiscard]] static Device cpu() noexcept;
    [[nodiscard]] static Device opencl(std::size_t platform_index = 0,
                                       std::size_t device_index = 0);

    [[nodiscard]] DeviceType type() const noexcept;
    [[nodiscard]] std::size_t platform_index() const noexcept;
    [[nodiscard]] std::size_t device_index() const noexcept;
    [[nodiscard]] std::string name() const;
    void synchronize() const;
    void set_profiling(bool enabled) const;
    [[nodiscard]] std::vector<KernelProfile> profile(bool reset = true) const;
    [[nodiscard]] OpenCLRuntimeStatistics runtime_statistics() const;
    void clear_memory_pool() const;

    friend bool operator==(const Device&, const Device&) = default;

  private:
    Device(DeviceType type, std::size_t platform_index, std::size_t device_index) noexcept;
    DeviceType type_;
    std::size_t platform_index_;
    std::size_t device_index_;
};

[[nodiscard]] bool opencl_available() noexcept;

} // namespace clnn
