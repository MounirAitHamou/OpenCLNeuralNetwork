#pragma once

#include "Utils/OpenCLSetup.hpp"
#include "Utils/OptimizerType.hpp"
#include <string>
#include <iostream>
#include <vector>
#include <map>

class Optimizer {
public:
    
    float learning_rate;
    float weight_decay_rate;

    cl::CommandQueue queue;
    cl::Program program;
    cl::Context context;

    Optimizer(const OpenCLSetup& ocl_setup,
              float learning_rate,
              float weight_decay_rate)
        : queue(ocl_setup.queue), program(ocl_setup.program), context(ocl_setup.context),
          learning_rate(learning_rate), weight_decay_rate(weight_decay_rate) {}

    Optimizer(const OpenCLSetup& ocl_setup)
        : queue(ocl_setup.queue), program(ocl_setup.program), context(ocl_setup.context) {}

    virtual ~Optimizer() = default;

    virtual void updateParameters(std::string param_id,
                                  cl::Buffer& params_buf,
                                  cl::Buffer& grads_buf,
                                  size_t num_elements) = 0;

    virtual void step() {}

    virtual void print() const = 0;

    void printCLBuffer(const cl::Buffer& buffer, size_t size, const std::string& label = "Buffer") {
        std::vector<float> host_data(size);
        // Enqueue a blocking read to transfer data from device to host
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(float) * size, host_data.data());
        std::cout << label << " Buffer Data: ";
        for (const auto& value : host_data) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
    
    virtual OptimizerType getType() const = 0;

    virtual void saveOptimizer(H5::Group& optimizer_group,
                                  const std::map<size_t, std::pair<size_t, size_t>>& moments_sizes) const = 0;

    void saveBuffer(const cl::Buffer& buffer, H5::Group& group, const std::string& name, size_t size) const{
        if (H5Lexists(group.getId(), name.c_str(), H5P_DEFAULT)) {
            std::cerr << "Warning: Dataset '" << name << "' already exists. Skipping write.\n";
            return;
        }
        std::vector<float> host_data(size);
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size * sizeof(float), host_data.data());

        H5::DataSpace dataspace(H5S_SIMPLE);
        hsize_t dims[1] = {size};
        dataspace.setExtentSimple(1, dims);

        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_FLOAT, dataspace);
        dataset.write(host_data.data(), H5::PredType::NATIVE_FLOAT);
    }
    
    cl::Buffer loadBuffer(const H5::Group& layer_group, const std::string& buffer_name, size_t size) {
        cl::Buffer buffer(context, CL_MEM_READ_WRITE, size * sizeof(float));
        std::vector<float> data(size);
        layer_group.openDataSet(buffer_name).read(data.data(), H5::PredType::NATIVE_FLOAT);
        queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size * sizeof(float), data.data());
        return buffer;
    }
};