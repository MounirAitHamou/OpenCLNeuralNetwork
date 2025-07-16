#pragma once

#include <CL/opencl.hpp>
#include <H5Cpp.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <algorithm>

struct OpenCLSetup {

    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    OpenCLSetup();

    OpenCLSetup(cl::Context context, cl::CommandQueue queue, cl::Program program);

    static OpenCLSetup createOpenCLSetup(const std::string& kernelsPath = "kernels");

    static std::vector<std::string> getAllKernelFiles(const std::string& folderPath);
};