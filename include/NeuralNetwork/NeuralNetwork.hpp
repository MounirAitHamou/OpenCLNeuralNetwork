#pragma once

#include "Utils/NetworkConfig.hpp"
#include "DataProcessor/AllDataProcessors.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>

class NeuralNetwork {
public:
   
    std::vector<std::unique_ptr<Layer>> layers;

    size_t batch_size;
    Dimensions input_dims;

    OpenCLSetup ocl_setup;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    std::unique_ptr<Optimizer> optimizer;

    LossFunctionType loss_function_type;

    NeuralNetwork() = default;

    NeuralNetwork(const OpenCLSetup& ocl_setup, NetworkConfig::NetworkArgs network_args);
    NeuralNetwork(const OpenCLSetup& ocl_setup, const H5::H5File& file);

    int findNextTrainableLayer(int start_idx) const;

    cl::Buffer forward(const cl::Buffer& initial_input_batch, size_t current_batch_actual_size);
    void backprop(const cl::Buffer& original_network_input_batch, const cl::Buffer& target_batch_buf);
    void train(DataProcessor& dataProcessor, int epochs);


    NeuralNetwork& addDense(const size_t output_dims_int, ActivationType activation_type);

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

    void saveNetwork(const std::string& filename) const;
    
    static NeuralNetwork loadNetwork(const OpenCLSetup& ocl_setup, const std::string& file_name);
};