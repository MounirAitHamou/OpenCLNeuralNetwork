#include <CL/opencl.hpp>
#include "OpenCLSetup.hpp"
#include "LayerGPU.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>

class NeuralNetwork {
public:
    std::vector<LayerGPU> layers;
    float learning_rate;
    int batch_size;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    NeuralNetwork() = default;
    NeuralNetwork(OpenCLSetup ocl_setup, const int input_size, const std::vector<int>& hidden_layers_sizes, const int output_size, int batch_size, float learning_rate);

    void initialize();

    cl::Buffer forward(const cl::Buffer& initial_input_batch);
    void backprop(const cl::Buffer& original_network_input_batch, const cl::Buffer& target_batch_buf);
    void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs);

    bool equalNetwork(const NeuralNetwork& other) const;

    int save(const std::string& filename) const;
    static NeuralNetwork load(OpenCLSetup ocl_setup, const std::string& filename);
};