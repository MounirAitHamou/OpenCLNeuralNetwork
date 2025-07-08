#include "Optimizer/Adam/AdamOptimizer.hpp"

void AdamOptimizer::updateParameters(cl::Buffer& params_buf,
                                     cl::Buffer& grads_buf,
                                     size_t num_elements) {
    auto it = moment_buffers.find(&params_buf);
    cl::Buffer m_buf;
    cl::Buffer v_buf;

    if (it == moment_buffers.end()) {
        std::vector<float> zero_data(num_elements, 0.0f);

        m_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           num_elements * sizeof(float), zero_data.data());

        v_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           num_elements * sizeof(float), zero_data.data());

        moment_buffers[&params_buf] = {m_buf, v_buf};
    } else {
        m_buf = it->second.first;
        v_buf = it->second.second;
    }

    cl::Kernel kernel(program, "adam_update_parameters");

    kernel.setArg(0, params_buf);
    kernel.setArg(1, grads_buf);
    kernel.setArg(2, m_buf);
    kernel.setArg(3, v_buf);
    kernel.setArg(4, learning_rate);
    kernel.setArg(5, beta1);
    kernel.setArg(6, beta2);
    kernel.setArg(7, epsilon); 
    kernel.setArg(8, (int)num_elements);
    kernel.setArg(9, weight_decay_rate);
    // Calculate beta1^t and beta2^t on the host and pass them to the kernel for bias correction. This calculation is more efficient on the CPU.
    kernel.setArg(10, pow(beta1, (float)t));
    kernel.setArg(11, pow(beta2, (float)t));

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(num_elements), cl::NullRange);
}

void AdamOptimizer::step() {
    t++;
}