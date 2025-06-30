#pragma once
#include <CL/opencl.hpp>
#include "OpenCLSetup.hpp"
#include <vector>
#include <random>

struct LayerGPU {
    int input_size, output_size, batch_size;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Buffer weights, biases, outputs, deltas;

    LayerGPU(OpenCLSetup ocl_setup, int in_size, int out_size, int batch_sz);
    void setRandomParams();
    void runForward(const cl::Buffer& input);
};