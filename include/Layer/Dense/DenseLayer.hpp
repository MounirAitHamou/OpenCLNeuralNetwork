#pragma once

#include "Layer/Layer.hpp"

/**
 * @class DenseLayer
 * @brief Implements a fully connected (dense) layer in a neural network.
 *
 * This layer connects every neuron in its input to every neuron in its output.
 * It performs a matrix multiplication of the input with a weight matrix and
 * adds a bias vector, followed by an activation function.
 */
class DenseLayer : public Layer {
public:
    /**
     * @brief Constructs a new DenseLayer object.
     *
     * @param ocl_setup A reference to the OpenCLSetup object, providing access to OpenCL context,
     * device, and command queue for GPU computations.
     * @param input_dims The dimensions of the input data to this layer.
     * @param output_dims The dimensions of the output data from this layer.
     * @param act_type The type of activation function to use for this layer (default: ReLU).
     * @param batch_size The number of samples processed in one forward/backward pass (default: 1).
     */
    DenseLayer(const OpenCLSetup& ocl_setup,
               const Dimensions& input_dims, const Dimensions& output_dims,
               ActivationType act_type = ActivationType::ReLU, size_t batch_size = 1);

    /**
     * @brief Destroys the DenseLayer object.
     *
     * Uses default destructor behavior as no special resource deallocation is needed
     * beyond what smart pointers or OpenCL handles manage.
     */
    ~DenseLayer() = default;

    /**
     * @brief Initializes the weights and biases of the dense layer.
     *
     * This method is an override from the base Layer class and is responsible
     * for setting up the initial values of the weight matrix and bias vector,
     * using a random initialization strategy.
     */
    void initializeWeightsAndBiases() override;

    /**
     * @brief Performs the forward pass computation for the dense layer.
     *
     * @param input_buffer An OpenCL buffer containing the input data from the previous layer.
     * The output of this layer is stored internally in an OpenCL buffer
     * (e.g., `output_buffer_`) which can be accessed by subsequent layers.
     */
    void runForward(const cl::Buffer& input_buffer) override;

    /**
     * @brief Computes the deltas (error gradients) for the output layer.
     *
     * This method is specifically used for the *output layer* of the network
     * to calculate the initial deltas based on the difference between predictions
     * and true labels.
     *
     * @param true_labels_buffer An OpenCL buffer containing the true labels for the current batch.
     * @param loss_function_type The type of loss function used to compute the error
     * (default: Mean Squared Error).
     */
    void computeOutputDeltas(const cl::Buffer& true_labels_buffer, const LossFunctionType& loss_function_type = LossFunctionType::MeanSquaredError) override;

    /**
     * @brief Backpropagates deltas from the next layer to the current dense layer.
     *
     * This method calculates the deltas for the current layer based on the
     * deltas of the subsequent layer and the weights connecting them.
     *
     * @param next_layer_weights An OpenCL buffer containing the weights of the next layer.
     * @param next_layer_deltas An OpenCL buffer containing the deltas of the next layer.
     * @param next_layer_output_size The output size (number of neurons) of the next layer.
     */
    void backpropDeltas(const cl::Buffer& next_layer_weights, const cl::Buffer& next_layer_deltas, const size_t next_layer_output_size) override;

    /**
     * @brief Calculates the gradients for the weights of the dense layer.
     *
     * This method computes how much each weight contributes to the total error,
     * based on the input to the current layer and the deltas of the current layer.
     *
     * @param inputs_to_current_layer An OpenCL buffer containing the input activations
     * from the previous layer, which are the inputs to this layer.
     */
    void calculateWeightGradients(const cl::Buffer& inputs_to_current_layer) override;

    /**
     * @brief Calculates the gradients for the biases of the dense layer.
     *
     * This method computes how much each bias contributes to the total error,
     * based on the deltas of the current layer.
     */
    void calculateBiasGradients() override;
};