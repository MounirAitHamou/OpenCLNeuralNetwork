#pragma once
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>

struct OpenCLSetup {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    OpenCLSetup();
    OpenCLSetup(cl::Context context, cl::CommandQueue queue, cl::Program program);
    static OpenCLSetup createOpenCLSetup(const std::string& kernel_file = "kernels/kernels.cl");
};