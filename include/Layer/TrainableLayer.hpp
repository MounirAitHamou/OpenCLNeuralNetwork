#pragma once

#include "Layer/Layer.hpp"

class TrainableLayer : public Layer {
public:
    cl::Buffer weights;
    cl::Buffer biases;
    cl::Buffer weight_gradients;
    cl::Buffer bias_gradients;

    TrainableLayer(size_t layer_id,
                   const OpenCLSetup& ocl_setup,
                   const Dimensions& input_dims,
                   const Dimensions& output_dims,
                   size_t batch_size = 1)
        : Layer(layer_id, ocl_setup, input_dims, output_dims, batch_size)
    {allocateParameterBuffers();}

    TrainableLayer(const OpenCLSetup& ocl_setup, const size_t batch_size)
        : Layer(ocl_setup, batch_size) {}

    
    virtual void calculateWeightGradients(const cl::Buffer& inputs_to_current_layer) = 0;
    virtual void calculateBiasGradients() = 0;

    bool isTrainable() const override { return true; }

    cl::Buffer& getWeights() { return weights; }
    cl::Buffer& getBiases() { return biases; }

    cl::Buffer& getWeightGradients() { return weight_gradients; }
    cl::Buffer& getBiasGradients() { return bias_gradients; }

    virtual size_t getWeightsSize() const {
        return input_dims.getTotalElements() * output_dims.getTotalElements();
    }

    virtual size_t getBiasesSize() const {
        return output_dims.getTotalElements();
    }
    
    virtual void initializeWeightsAndBiases() = 0;

    virtual ~TrainableLayer() = default;

    protected:
    void allocateParameterBuffers() {
        size_t flat_input_size = input_dims.getTotalElements();
        size_t flat_output_size = output_dims.getTotalElements();

        weights          = cl::Buffer(context, CL_MEM_READ_WRITE, flat_input_size * flat_output_size* sizeof(float));
        weight_gradients = cl::Buffer(context, CL_MEM_READ_WRITE, flat_input_size * flat_output_size * sizeof(float));

        biases           = cl::Buffer(context, CL_MEM_READ_WRITE, flat_output_size * sizeof(float));
        bias_gradients   = cl::Buffer(context, CL_MEM_READ_WRITE, flat_output_size * sizeof(float));
    }
};