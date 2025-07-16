#pragma once

#include "Utils/Dimensions.hpp"
#include "Utils/ActivationType.hpp"
#include "Utils/OpenCLSetup.hpp"
#include "Utils/LossFunctionType.hpp"
#include "Utils/LayerType.hpp"

#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <typeinfo>

class Layer;

class Layer {
public:

    size_t layer_id;

    Dimensions input_dims;
    Dimensions output_dims;

    size_t batch_size;

    cl::Buffer outputs;
    cl::Buffer deltas;

    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    Layer(const size_t layer_id, 
          const OpenCLSetup& ocl_setup,
          const Dimensions& input_dims, const Dimensions& output_dims, size_t batch_size = 1)
        : layer_id(layer_id), context(ocl_setup.context), queue(ocl_setup.queue), program(ocl_setup.program),
          input_dims(input_dims), output_dims(output_dims), batch_size(batch_size) {allocateBuffers();}

    Layer(const OpenCLSetup& ocl_setup, const size_t batch_size)
        : context(ocl_setup.context), queue(ocl_setup.queue), program(ocl_setup.program), batch_size(batch_size) {}

    virtual ~Layer() = default;

    virtual void runForward(const cl::Buffer& input_buffer) = 0;
    virtual void computeOutputDeltas(const cl::Buffer& true_labels_buffer, const LossFunctionType& loss_function_type = LossFunctionType::MeanSquaredError) = 0;
    virtual void backpropDeltas(const cl::Buffer& next_layer_deltas, const cl::Buffer* next_layer_weights_ptr, const size_t next_layer_output_size) = 0;
    
    virtual bool isTrainable() const { return false; }
    cl::Buffer& getOutputs() {
        return outputs;
    }
    cl::Buffer& getDeltas() {
        return deltas;
    }
    virtual cl::Buffer& getWeights() { throw std::runtime_error("No weights"); }
    virtual cl::Buffer& getBiases() { throw std::runtime_error("No biases"); }
    virtual cl::Buffer& getWeightGradients() { throw std::runtime_error("No weight gradients"); }
    virtual cl::Buffer& getBiasGradients() { throw std::runtime_error("No bias gradients"); }
    virtual void setBatchSize(size_t new_batch_size) {
        batch_size = new_batch_size;
    }

    virtual size_t getTotalOutputElements() const {
        return output_dims.getTotalElements();
    }

    virtual size_t getTotalInputElements() const {
        return input_dims.getTotalElements();
    }

    float getRandomWeight(float min, float max) {
        static std::default_random_engine generator(
            static_cast<unsigned int>(
                std::chrono::system_clock::now().time_since_epoch().count()
            )
        );
        std::uniform_real_distribution<float> distribution(min, max);
        return distribution(generator);
    }

    virtual void print() const = 0;

    void printCLBuffer(const cl::Buffer& buffer, size_t size, const std::string& label = "Buffer") const {
        std::vector<float> host_data(size);
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(float) * size, host_data.data());
        std::cout << label << " Buffer Data: ";
        for (const auto& value : host_data) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }

    virtual LayerType getType() const = 0;

    virtual void saveLayer(H5::Group& layer_group) const = 0;

    void saveBuffer(const cl::Buffer& buffer, H5::Group& group, const std::string& name, const size_t size) const {
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
    protected:
    void allocateBuffers() {
        size_t flat_input_size = input_dims.getTotalElements();
        size_t flat_output_size = output_dims.getTotalElements();

        outputs = cl::Buffer(context, CL_MEM_READ_WRITE, batch_size * flat_output_size * sizeof(float));
        deltas  = cl::Buffer(context, CL_MEM_READ_WRITE, batch_size * flat_output_size * sizeof(float));
    }
};