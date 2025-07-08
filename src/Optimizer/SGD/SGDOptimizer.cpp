#include "Optimizer/SGD/SGDOptimizer.hpp"

void SGDOptimizer::updateParameters(cl::Buffer& params_buf,
                                     cl::Buffer& grads_buf,
                                     size_t num_elements) {
    cl::Kernel kernel(program, "sgd_update_parameters");

    kernel.setArg(0, params_buf);
    kernel.setArg(1, grads_buf);
    kernel.setArg(2, (cl_uint)num_elements);
    kernel.setArg(3, learning_rate);
    kernel.setArg(4, weight_decay_rate);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(num_elements), cl::NullRange);
}