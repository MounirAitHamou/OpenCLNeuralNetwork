#pragma once

#include <cstddef>
#include <cstdint>

namespace clnn::opencl::api {

using Int = std::int32_t;
using UInt = std::uint32_t;
using ULong = std::uint64_t;
using Bitfield = ULong;
using DeviceType = Bitfield;
using MemoryFlags = Bitfield;
using Bool = UInt;
using ContextProperty = std::intptr_t;
using QueueProperty = Bitfield;
using PlatformInfo = UInt;
using DeviceInfo = UInt;
using ProgramBuildInfo = UInt;
using EventInfo = UInt;
using ProfilingInfo = UInt;

struct PlatformId;
struct DeviceId;
struct Context;
struct CommandQueue;
struct Memory;
struct Program;
struct Kernel;
struct Event;

using PlatformHandle = PlatformId*;
using DeviceHandle = DeviceId*;
using ContextHandle = Context*;
using QueueHandle = CommandQueue*;
using MemoryHandle = Memory*;
using ProgramHandle = Program*;
using KernelHandle = Kernel*;
using EventHandle = Event*;

inline constexpr Int success = 0;
inline constexpr Bool false_value = 0;
inline constexpr Bool true_value = 1;
inline constexpr DeviceType device_type_gpu = 1ULL << 2U;
inline constexpr MemoryFlags memory_read_write = 1ULL << 0U;
inline constexpr MemoryFlags memory_copy_host_pointer = 1ULL << 5U;
inline constexpr QueueProperty queue_profiling_enable = 1ULL << 1U;
inline constexpr PlatformInfo platform_name = 0x0902;
inline constexpr DeviceInfo device_name = 0x102B;
inline constexpr DeviceInfo device_local_memory_size = 0x1023;
inline constexpr ProgramBuildInfo program_build_log = 0x1183;
inline constexpr EventInfo event_command_execution_status = 0x11D3;
inline constexpr ProfilingInfo profiling_command_start = 0x1282;
inline constexpr ProfilingInfo profiling_command_end = 0x1283;
inline constexpr Int complete = 0;

class Functions final {
  public:
    Functions();
    ~Functions();
    Functions(const Functions&) = delete;
    Functions& operator=(const Functions&) = delete;

    Int (*get_platform_ids)(UInt, PlatformHandle*, UInt*);
    Int (*get_platform_info)(PlatformHandle, PlatformInfo, std::size_t, void*, std::size_t*);
    Int (*get_device_ids)(PlatformHandle, DeviceType, UInt, DeviceHandle*, UInt*);
    Int (*get_device_info)(DeviceHandle, DeviceInfo, std::size_t, void*, std::size_t*);
    ContextHandle (*create_context)(const ContextProperty*, UInt, const DeviceHandle*,
                                    void (*)(const char*, const void*, std::size_t, void*), void*,
                                    Int*);
    Int (*release_context)(ContextHandle);
    QueueHandle (*create_command_queue)(ContextHandle, DeviceHandle, QueueProperty, Int*);
    Int (*release_command_queue)(QueueHandle);
    MemoryHandle (*create_buffer)(ContextHandle, MemoryFlags, std::size_t, void*, Int*);
    Int (*release_memory_object)(MemoryHandle);
    ProgramHandle (*create_program_with_source)(ContextHandle, UInt, const char**,
                                                const std::size_t*, Int*);
    Int (*build_program)(ProgramHandle, UInt, const DeviceHandle*, const char*,
                         void (*)(ProgramHandle, void*), void*);
    Int (*get_program_build_info)(ProgramHandle, DeviceHandle, ProgramBuildInfo, std::size_t, void*,
                                  std::size_t*);
    Int (*release_program)(ProgramHandle);
    KernelHandle (*create_kernel)(ProgramHandle, const char*, Int*);
    Int (*release_kernel)(KernelHandle);
    Int (*set_kernel_argument)(KernelHandle, UInt, std::size_t, const void*);
    Int (*enqueue_write_buffer)(QueueHandle, MemoryHandle, Bool, std::size_t, std::size_t,
                                const void*, UInt, const EventHandle*, EventHandle*);
    Int (*enqueue_read_buffer)(QueueHandle, MemoryHandle, Bool, std::size_t, std::size_t, void*,
                               UInt, const EventHandle*, EventHandle*);
    Int (*enqueue_nd_range_kernel)(QueueHandle, KernelHandle, UInt, const std::size_t*,
                                   const std::size_t*, const std::size_t*, UInt, const EventHandle*,
                                   EventHandle*);
    Int (*get_event_info)(EventHandle, EventInfo, std::size_t, void*, std::size_t*);
    Int (*get_event_profiling_info)(EventHandle, ProfilingInfo, std::size_t, void*, std::size_t*);
    Int (*release_event)(EventHandle);
    Int (*flush)(QueueHandle);
    Int (*finish)(QueueHandle);

  private:
    void* library_ = nullptr;
};

} // namespace clnn::opencl::api
