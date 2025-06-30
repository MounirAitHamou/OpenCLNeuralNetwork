#include "LayerGPU.hpp"

LayerGPU::LayerGPU(OpenCLSetup ocl_setup, int in_size, int out_size, int batch_sz)
    : queue(ocl_setup.queue), program(ocl_setup.program), input_size(in_size), output_size(out_size), batch_size(batch_sz)
        
{
    weights     = cl::Buffer(ocl_setup.context, CL_MEM_READ_WRITE, sizeof(float) * input_size * output_size);
    biases      = cl::Buffer(ocl_setup.context, CL_MEM_READ_WRITE, sizeof(float) * output_size);
    outputs     = cl::Buffer(ocl_setup.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * output_size);
    deltas      = cl::Buffer(ocl_setup.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * output_size);
}

void LayerGPU::setRandomParams() {
    std::vector<float> host_weights(input_size * output_size);
    std::vector<float> host_biases(output_size);
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& w : host_weights) w = dist(gen);
    for (auto& b : host_biases) b = dist(gen);

    queue.enqueueWriteBuffer(weights, CL_TRUE, 0, sizeof(float) * host_weights.size(), host_weights.data());
    queue.enqueueWriteBuffer(biases,  CL_TRUE, 0, sizeof(float) * host_biases.size(),  host_biases.data());
}

void LayerGPU::runForward(const cl::Buffer& input) {
    cl::Kernel kernel(program, "layer_forward_batch");
    
    kernel.setArg(0, input);
    kernel.setArg(1, weights);
    kernel.setArg(2, biases);
    kernel.setArg(3, outputs);
    kernel.setArg(4, input_size);
    kernel.setArg(5, output_size);
    kernel.setArg(6, batch_size);

    cl::NDRange global(output_size, batch_size);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    queue.finish();
}