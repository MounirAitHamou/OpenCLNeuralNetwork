#include "Optimizer/SGD/SGDOptimizer.hpp"

SGDOptimizer::SGDOptimizer(const OpenCLSetup& ocl_setup, const H5::Group& optimizer_group)
    :Optimizer(ocl_setup){
    optimizer_group.openAttribute("learning_rate").read(H5::PredType::NATIVE_FLOAT, &learning_rate);
    optimizer_group.openAttribute("weight_decay_rate").read(H5::PredType::NATIVE_FLOAT, &weight_decay_rate);
    }

void SGDOptimizer::updateParameters(std::string param_id,
                                     cl::Buffer& params_buf,
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

void SGDOptimizer::saveOptimizer(H5::Group& optimizer_group,
                                  const std::map<size_t, std::pair<size_t, size_t>>& moments_sizes) const {
    H5::DataSpace scalar_dataspace(H5S_SCALAR);
    optimizer_group.createAttribute("learning_rate", H5::PredType::NATIVE_FLOAT, scalar_dataspace)
        .write(H5::PredType::NATIVE_FLOAT, &learning_rate);
    optimizer_group.createAttribute("weight_decay_rate", H5::PredType::NATIVE_FLOAT, scalar_dataspace)
        .write(H5::PredType::NATIVE_FLOAT, &weight_decay_rate);
}