#include "Optimizer/AdamW/AdamWOptimizer.hpp"

AdamWOptimizer::AdamWOptimizer(const OpenCLSetup& ocl_setup, const H5::Group& optimizer_group)
    :Optimizer(ocl_setup){

    optimizer_group.openAttribute("learning_rate").read(H5::PredType::NATIVE_FLOAT, &learning_rate);
    optimizer_group.openAttribute("weight_decay_rate").read(H5::PredType::NATIVE_FLOAT, &weight_decay_rate);
    optimizer_group.openAttribute("beta1").read(H5::PredType::NATIVE_FLOAT, &beta1);
    optimizer_group.openAttribute("beta2").read(H5::PredType::NATIVE_FLOAT, &beta2);
    optimizer_group.openAttribute("epsilon").read(H5::PredType::NATIVE_FLOAT, &epsilon);
    optimizer_group.openAttribute("t").read(H5::PredType::NATIVE_UINT, &t);
    
    

    H5::Group moment_buffers_group = optimizer_group.openGroup("moment_buffers");
    
    hsize_t num_layers = moment_buffers_group.getNumObjs();
    for (hsize_t i = 0; i < num_layers; ++i) {
        std::string layer_name = moment_buffers_group.getObjnameByIdx(i);
        H5::Group layer_group = moment_buffers_group.openGroup(layer_name);
        size_t layer_id;
        layer_group.openAttribute("layer_id").read(H5::PredType::NATIVE_HSIZE, &layer_id);

        size_t weights_size, biases_size;
        layer_group.openAttribute("weights_size").read(H5::PredType::NATIVE_HSIZE, &weights_size);
        layer_group.openAttribute("biases_size").read(H5::PredType::NATIVE_HSIZE, &biases_size);
        if (weights_size > 0) {
            std::string weight_key = std::to_string(layer_id) + "_weights";
            cl::Buffer m_buf = loadBuffer(layer_group, "weights_first_moment_buffer", weights_size);
            cl::Buffer v_buf = loadBuffer(layer_group, "weights_second_moment_buffer", weights_size);
            moment_buffers[weight_key] = {m_buf, v_buf};
        }

        if (biases_size > 0) {
            std::string bias_key = std::to_string(layer_id) + "_biases";
            cl::Buffer m_buf = loadBuffer(layer_group, "biases_first_moment_buffer", biases_size);
            cl::Buffer v_buf = loadBuffer(layer_group, "biases_second_moment_buffer", biases_size);
            moment_buffers[bias_key] = {m_buf, v_buf};
        }
    }
}

void AdamWOptimizer::updateParameters(std::string param_id,
                                     cl::Buffer& params_buf,
                                     cl::Buffer& grads_buf,
                                     size_t num_elements) {
    auto it = moment_buffers.find(param_id);
    cl::Buffer m_buf;
    cl::Buffer v_buf;

    if (it == moment_buffers.end()) {
        std::vector<float> zero_data(num_elements, 0.0f);

        m_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           num_elements * sizeof(float), zero_data.data());

        v_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           num_elements * sizeof(float), zero_data.data());

        moment_buffers[param_id] = {m_buf, v_buf};
    } else {
        m_buf = it->second.first;
        v_buf = it->second.second;
    }

    cl::Kernel kernel(program, "adamw_update_parameters");

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
    kernel.setArg(10, pow(beta1, (float)t));
    kernel.setArg(11, pow(beta2, (float)t));

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(num_elements), cl::NullRange);
}

void AdamWOptimizer::step() {
    t++;
}

void AdamWOptimizer::saveOptimizer(H5::Group& optimizer_group,
                                  const std::map<size_t, std::pair<size_t, size_t>>& moments_sizes) const {
    H5::DataSpace scalar_dataspace(H5S_SCALAR);
    optimizer_group.createAttribute("learning_rate", H5::PredType::NATIVE_FLOAT, scalar_dataspace)
        .write(H5::PredType::NATIVE_FLOAT, &learning_rate);
    optimizer_group.createAttribute("weight_decay_rate", H5::PredType::NATIVE_FLOAT, scalar_dataspace)
        .write(H5::PredType::NATIVE_FLOAT, &weight_decay_rate);
    optimizer_group.createAttribute("beta1", H5::PredType::NATIVE_FLOAT, scalar_dataspace)
        .write(H5::PredType::NATIVE_FLOAT, &beta1);
    optimizer_group.createAttribute("beta2", H5::PredType::NATIVE_FLOAT, scalar_dataspace)
        .write(H5::PredType::NATIVE_FLOAT, &beta2);
    optimizer_group.createAttribute("epsilon", H5::PredType::NATIVE_FLOAT, scalar_dataspace)
        .write(H5::PredType::NATIVE_FLOAT, &epsilon);
    optimizer_group.createAttribute("t", H5::PredType::NATIVE_UINT, scalar_dataspace)
        .write(H5::PredType::NATIVE_UINT, &t);

    H5::Group moment_buffers_group = optimizer_group.createGroup("moment_buffers");
    for (const auto& [layer_id, sizes] : moments_sizes) {
        size_t weights_size = sizes.first;
        size_t biases_size = sizes.second;
        std::string layer_id_str = std::to_string(layer_id);
        H5::Group layer_group = moment_buffers_group.createGroup(layer_id_str);
        layer_group.createAttribute("layer_id", H5::PredType::NATIVE_HSIZE, scalar_dataspace)
            .write(H5::PredType::NATIVE_HSIZE, &layer_id);
        layer_group.createAttribute("weights_size", H5::PredType::NATIVE_HSIZE, scalar_dataspace)
            .write(H5::PredType::NATIVE_HSIZE, &weights_size);
        layer_group.createAttribute("biases_size", H5::PredType::NATIVE_HSIZE, scalar_dataspace)
            .write(H5::PredType::NATIVE_HSIZE, &biases_size);
        if (weights_size > 0) {
            std::string weight_key = layer_id_str + "_weights";
            const cl::Buffer& m_buf = moment_buffers.at(weight_key).first;
            const cl::Buffer& v_buf = moment_buffers.at(weight_key).second;

            saveBuffer(m_buf, layer_group, "weights_first_moment_buffer", weights_size);
            saveBuffer(v_buf, layer_group, "weights_second_moment_buffer", weights_size);
        }

        if (biases_size > 0) {
            std::string bias_key = layer_id_str + "_biases";
            const cl::Buffer& m_buf = moment_buffers.at(bias_key).first;
            const cl::Buffer& v_buf = moment_buffers.at(bias_key).second;

            saveBuffer(m_buf, layer_group, "biases_first_moment_buffer", biases_size);
            saveBuffer(v_buf, layer_group, "biases_second_moment_buffer", biases_size);
        }
    }
}