#include "opencl_api.hpp"

#include <stdexcept>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace clnn::opencl::api {
namespace {

void* open_library() {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryW(L"OpenCL.dll"));
#elif defined(__APPLE__)
    return dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_NOW | RTLD_LOCAL);
#else
    if (auto* handle = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL))
        return handle;
    return dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_library(void* library) {
#ifdef _WIN32
    if (library != nullptr)
        FreeLibrary(reinterpret_cast<HMODULE>(library));
#else
    if (library != nullptr)
        dlclose(library);
#endif
}

void* load_symbol(void* library, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

template <typename Function> Function load(void* library, const char* name) {
    const auto symbol = load_symbol(library, name);
    if (symbol == nullptr) {
        throw std::runtime_error(std::string("OpenCL loader does not export ") + name);
    }
    return reinterpret_cast<Function>(symbol);
}

} // namespace

Functions::Functions() : library_(open_library()) {
    if (library_ == nullptr) {
        throw std::runtime_error("OpenCL loader library was not found");
    }
    try {
        get_platform_ids = load<decltype(get_platform_ids)>(library_, "clGetPlatformIDs");
        get_platform_info = load<decltype(get_platform_info)>(library_, "clGetPlatformInfo");
        get_device_ids = load<decltype(get_device_ids)>(library_, "clGetDeviceIDs");
        get_device_info = load<decltype(get_device_info)>(library_, "clGetDeviceInfo");
        create_context = load<decltype(create_context)>(library_, "clCreateContext");
        release_context = load<decltype(release_context)>(library_, "clReleaseContext");
        create_command_queue =
            load<decltype(create_command_queue)>(library_, "clCreateCommandQueue");
        release_command_queue =
            load<decltype(release_command_queue)>(library_, "clReleaseCommandQueue");
        create_buffer = load<decltype(create_buffer)>(library_, "clCreateBuffer");
        release_memory_object =
            load<decltype(release_memory_object)>(library_, "clReleaseMemObject");
        create_program_with_source =
            load<decltype(create_program_with_source)>(library_, "clCreateProgramWithSource");
        build_program = load<decltype(build_program)>(library_, "clBuildProgram");
        get_program_build_info =
            load<decltype(get_program_build_info)>(library_, "clGetProgramBuildInfo");
        release_program = load<decltype(release_program)>(library_, "clReleaseProgram");
        create_kernel = load<decltype(create_kernel)>(library_, "clCreateKernel");
        release_kernel = load<decltype(release_kernel)>(library_, "clReleaseKernel");
        set_kernel_argument = load<decltype(set_kernel_argument)>(library_, "clSetKernelArg");
        enqueue_write_buffer =
            load<decltype(enqueue_write_buffer)>(library_, "clEnqueueWriteBuffer");
        enqueue_read_buffer = load<decltype(enqueue_read_buffer)>(library_, "clEnqueueReadBuffer");
        enqueue_nd_range_kernel =
            load<decltype(enqueue_nd_range_kernel)>(library_, "clEnqueueNDRangeKernel");
        get_event_info = load<decltype(get_event_info)>(library_, "clGetEventInfo");
        get_event_profiling_info =
            load<decltype(get_event_profiling_info)>(library_, "clGetEventProfilingInfo");
        release_event = load<decltype(release_event)>(library_, "clReleaseEvent");
        flush = load<decltype(flush)>(library_, "clFlush");
        finish = load<decltype(finish)>(library_, "clFinish");
    } catch (...) {
        close_library(library_);
        library_ = nullptr;
        throw;
    }
}

Functions::~Functions() {
    close_library(library_);
}

} // namespace clnn::opencl::api
