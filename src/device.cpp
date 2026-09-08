#include "clnn/device.hpp"

#include "opencl/backend.hpp"

namespace clnn {

Device::Device(const DeviceType type, const std::size_t platform_index,
               const std::size_t device_index) noexcept
    : type_(type), platform_index_(platform_index), device_index_(device_index) {}

Device Device::cpu() noexcept {
    return Device(DeviceType::cpu, 0, 0);
}

Device Device::opencl(const std::size_t platform_index, const std::size_t device_index) {
    return Device(DeviceType::opencl, platform_index, device_index);
}

DeviceType Device::type() const noexcept {
    return type_;
}
std::size_t Device::platform_index() const noexcept {
    return platform_index_;
}
std::size_t Device::device_index() const noexcept {
    return device_index_;
}

std::string Device::name() const {
    return type_ == DeviceType::cpu ? "CPU" : opencl::device_name(*this);
}

void Device::synchronize() const {
    if (type_ == DeviceType::opencl)
        opencl::synchronize(*this);
}

void Device::set_profiling(const bool enabled) const {
    if (type_ == DeviceType::opencl)
        opencl::set_profiling(*this, enabled);
}

std::vector<KernelProfile> Device::profile(const bool reset) const {
    return type_ == DeviceType::opencl ? opencl::profile(*this, reset)
                                       : std::vector<KernelProfile>{};
}

OpenCLRuntimeStatistics Device::runtime_statistics() const {
    return type_ == DeviceType::opencl ? opencl::runtime_statistics(*this)
                                       : OpenCLRuntimeStatistics{};
}

void Device::clear_memory_pool() const {
    if (type_ == DeviceType::opencl)
        opencl::clear_memory_pool(*this);
}

bool opencl_available() noexcept {
    return opencl::available();
}

} // namespace clnn
