#include "Layer/Dense/DenseLayer.hpp"

DenseLayer::DenseLayer(const OpenCLSetup& ocl_setup,
                       const Dimensions& input_dims, const Dimensions& output_dims,
                       ActivationType act_type, size_t batch_size)
    : Layer(ocl_setup, input_dims, output_dims, act_type, batch_size) {
    size_t flat_input_size = input_dims.getTotalElements();
    size_t flat_output_size = output_dims.getTotalElements();

    weights = cl::Buffer(context, CL_MEM_READ_WRITE, flat_input_size * flat_output_size * sizeof(float));
    biases = cl::Buffer(context, CL_MEM_READ_WRITE, flat_output_size * sizeof(float));
    pre_activations = cl::Buffer(context, CL_MEM_READ_WRITE, batch_size * flat_output_size * sizeof(float));
    outputs = cl::Buffer(context, CL_MEM_READ_WRITE, batch_size * flat_output_size * sizeof(float));
    deltas = cl::Buffer(context, CL_MEM_READ_WRITE, batch_size * flat_output_size * sizeof(float));
    weight_gradients = cl::Buffer(context, CL_MEM_READ_WRITE, flat_input_size * flat_output_size * sizeof(float));
    bias_gradients = cl::Buffer(context, CL_MEM_READ_WRITE, flat_output_size * sizeof(float));

    initializeWeightsAndBiases();
}

void DenseLayer::initializeWeightsAndBiases() {
    size_t flat_input_size = input_dims.getTotalElements();
    size_t flat_output_size = output_dims.getTotalElements();

    std::vector<float> h_weights(flat_input_size * flat_output_size);
    std::vector<float> h_biases(flat_output_size);

    // Calculate the limit for Xavier/Glorot uniform initialization.
    // This formula is sqrt(6 / (fan_in + fan_out)).
    float limit = std::sqrt(6.0f / (flat_input_size + flat_output_size));

    for (auto& weight : h_weights) {
        weight = getRandomWeight(-limit, limit);
    }
    // Initialize biases to a small random value.
    for (auto& bias : h_biases) {
        bias = getRandomWeight(-0.5f, 0.5f);
    }

    queue.enqueueWriteBuffer(weights, CL_TRUE, 0, h_weights.size() * sizeof(float), h_weights.data());
    queue.enqueueWriteBuffer(biases, CL_TRUE, 0, h_biases.size() * sizeof(float), h_biases.data());
}

void DenseLayer::runForward(const cl::Buffer& input_buffer) {
    size_t flat_input_size = input_dims.getTotalElements();
    size_t flat_output_size = output_dims.getTotalElements();

    cl::Kernel kernel(program, "dense_forward_batch");

    kernel.setArg(0, weights);
    kernel.setArg(1, biases);
    kernel.setArg(2, input_buffer);
    kernel.setArg(3, outputs);
    kernel.setArg(4, pre_activations);
    kernel.setArg(5, (cl_uint)flat_input_size);
    kernel.setArg(6, (cl_uint)flat_output_size);
    kernel.setArg(7, (cl_uint)batch_size);
    kernel.setArg(8, (cl_uint)activation_type);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size, batch_size), cl::NullRange);
}

void DenseLayer::computeOutputDeltas(const cl::Buffer& true_labels_buffer,
                                     const LossFunctionType& loss_function_type) {
    size_t flat_output_size = output_dims.getTotalElements();

    cl::Kernel kernel(program, "dense_compute_output_deltas_batch");

    kernel.setArg(0, pre_activations);
    kernel.setArg(1, outputs);
    kernel.setArg(2, true_labels_buffer);
    kernel.setArg(3, deltas);
    kernel.setArg(4, (cl_uint)flat_output_size);
    kernel.setArg(5, (cl_uint)batch_size);
    kernel.setArg(6, (cl_uint)activation_type);
    kernel.setArg(7, (cl_uint)loss_function_type);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size, batch_size), cl::NullRange);
}

void DenseLayer::backpropDeltas(const cl::Buffer& next_layer_weights, const cl::Buffer& next_layer_deltas,
                               const size_t next_layer_output_size) {
    size_t flat_output_size = output_dims.getTotalElements();

    cl::Kernel kernel(program, "dense_backprop_deltas_batch");

    kernel.setArg(0, next_layer_weights);
    kernel.setArg(1, next_layer_deltas);
    kernel.setArg(2, deltas);
    kernel.setArg(3, pre_activations);
    kernel.setArg(4, (cl_uint)flat_output_size);
    kernel.setArg(5, (cl_uint)next_layer_output_size);
    kernel.setArg(6, (cl_uint)batch_size);
    kernel.setArg(7, (cl_uint)activation_type);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size, batch_size), cl::NullRange);
}

void DenseLayer::calculateWeightGradients(const cl::Buffer& inputs_to_current_layer) {
    size_t flat_input_size = input_dims.getTotalElements();
    size_t flat_output_size = output_dims.getTotalElements();

    cl::Kernel kernel(program, "dense_calculate_weight_gradients_batch");

    kernel.setArg(0, inputs_to_current_layer);
    kernel.setArg(1, deltas);
    kernel.setArg(2, weight_gradients);
    kernel.setArg(3, (cl_uint)flat_input_size);
    kernel.setArg(4, (cl_uint)flat_output_size);
    kernel.setArg(5, (cl_uint)batch_size);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size, flat_input_size), cl::NullRange);
}

void DenseLayer::calculateBiasGradients() {
    size_t flat_output_size = output_dims.getTotalElements();

    cl::Kernel kernel(program, "dense_calculate_bias_gradients_batch");

    kernel.setArg(0, deltas);
    kernel.setArg(1, bias_gradients);
    kernel.setArg(2, (cl_uint)flat_output_size);
    kernel.setArg(3, (cl_uint)batch_size);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size), cl::NullRange);
}