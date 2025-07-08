#pragma once

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <algorithm>

/**
 * @struct OpenCLSetup
 * @brief A utility structure to encapsulate the core OpenCL objects needed for GPU computation.
 *
 * This struct holds the OpenCL context, command queue, and program, which are
 * fundamental for executing OpenCL kernels. It also provides static methods for
 * easy initialization by discovering and compiling OpenCL kernel files from a specified directory.
 */
struct OpenCLSetup {
    /**
     * @brief The OpenCL context.
     * A context manages all OpenCL objects (devices, command queues, memory, programs, kernels)
     * and operations for a specific set of devices.
     */
    cl::Context context;

    /**
     * @brief The OpenCL command queue.
     * Commands (like kernel execution, memory transfers) are enqueued to this queue
     * for execution on the OpenCL device.
     */
    cl::CommandQueue queue;

    /**
     * @brief The OpenCL program object.
     * This object holds the compiled OpenCL kernels (functions written in OpenCL C)
     * that will be executed on the GPU.
     */
    cl::Program program;

    // --- Constructors ---

    /**
     * @brief Default constructor for OpenCLSetup.
     * Initializes member variables to their default (empty/null) states.
     * This constructor might be used if the setup is done in a separate method.
     */
    OpenCLSetup();

    /**
     * @brief Parameterized constructor for OpenCLSetup.
     * Allows direct initialization of the context, queue, and program.
     *
     * @param context The OpenCL context to use.
     * @param queue The OpenCL command queue to use.
     * @param program The OpenCL program to use.
     */
    OpenCLSetup(cl::Context context, cl::CommandQueue queue, cl::Program program);

    // --- Static Factory Methods ---

    /**
     * @brief Static factory method to create and initialize an OpenCLSetup object.
     *
     * This method automatically discovers available OpenCL platforms and devices,
     * creates a context and command queue, and then compiles all OpenCL kernel
     * files found in the specified `kernelsPath` into a single `cl::Program`.
     *
     * @param kernelsPath The path to the directory containing OpenCL kernel (.cl) files (default: "kernels").
     * @return An initialized `OpenCLSetup` object ready for use.
     * @throws std::runtime_error if no OpenCL platforms/devices are found, or if kernel compilation fails.
     */
    static OpenCLSetup createOpenCLSetup(const std::string& kernelsPath = "kernels");

    /**
     * @brief Static utility method to get all OpenCL kernel file paths from a specified folder.
     *
     * It iterates through the given directory and collects paths to all files
     * that typically contain OpenCL C code (e.g., files with a '.cl' extension).
     *
     * @param folderPath The path to the directory to search for kernel files.
     * @return A `std::vector` of `std::string` containing the full paths to the kernel files.
     * @throws std::runtime_error if the specified folder path does not exist or is not a directory.
     */
    static std::vector<std::string> getAllKernelFiles(const std::string& folderPath);
};