#pragma once

#include "Layer/TrainableLayer.hpp"

class DenseLayer : public TrainableLayer {
public:

    ActivationType activation_type;

    cl::Buffer pre_activations;

    DenseLayer(const size_t layer_id, const OpenCLSetup& ocl_setup,
               const Dimensions& input_dims, const Dimensions& output_dims,
               ActivationType act_type = ActivationType::ReLU, size_t batch_size = 1);

    DenseLayer(const OpenCLSetup& ocl_setup, const H5::Group& layer_group, const size_t batch_size);

    ~DenseLayer() = default;

    void initializeWeightsAndBiases() override;

    void runForward(const cl::Buffer& input_buffer) override;
    void computeOutputDeltas(const cl::Buffer& true_labels_buffer, const LossFunctionType& loss_function_type = LossFunctionType::MeanSquaredError) override;
    void backpropDeltas(const cl::Buffer& next_layer_deltas, const cl::Buffer* next_layer_weights_ptr, const size_t next_layer_output_size) override;
    void calculateWeightGradients(const cl::Buffer& inputs_to_current_layer) override;
    void calculateBiasGradients() override;

    cl::Buffer& getPreActivations() {
        return pre_activations;
    }

    LayerType getType() const override {
        return LayerType::Dense;
    }

    void saveLayer(H5::Group& layer_group) const override;

    void print() const override {
        std::cout << "Activation Type: " << static_cast<unsigned int>(activation_type) << "\n";
        std::cout << "Weights Size: " << getWeightsSize() << "\n";
        std::cout << "Biases Size: " << getBiasesSize() << "\n";
        printCLBuffer(weights, getWeightsSize(), "Weights");
        printCLBuffer(biases, getBiasesSize(), "Biases");
        printCLBuffer(pre_activations, batch_size * output_dims.getTotalElements(), "Pre-activations");
        printCLBuffer(outputs, batch_size * output_dims.getTotalElements(), "Outputs");
        printCLBuffer(deltas, batch_size * output_dims.getTotalElements(), "Deltas");
        printCLBuffer(weight_gradients, getWeightsSize(), "Weight Gradients");
        printCLBuffer(bias_gradients, getBiasesSize(), "Bias Gradients");
    }

    protected:
    void allocatePreActivationBuffer() {
        size_t flat_output_size = output_dims.getTotalElements();
        pre_activations = cl::Buffer(context, CL_MEM_READ_WRITE, batch_size * flat_output_size * sizeof(float));
    }
};