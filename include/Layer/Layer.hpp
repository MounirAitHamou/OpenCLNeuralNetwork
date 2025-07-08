#pragma once

#include "Utils/Dimensions.hpp"
#include "Utils/ActivationType.hpp"
#include "Utils/OpenCLSetup.hpp"
#include "Utils/LossFunctionType.hpp"
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <typeinfo>

/**
 * @class Layer
 * @brief An abstract base class defining the interface for all neural network layers.
 *
 * This class provides common properties and pure virtual functions that every
 * concrete neural network layer (e.g., DenseLayer) must implement.
 * It also manages OpenCL buffers for data transfer between host and device,
 * and provides utility methods for debugging and initialization.
 */
class Layer {
public:
    // --- Layer Properties ---
    /**
     * @brief The number of samples processed in one forward/backward pass.
     * This affects the size of data buffers and kernel execution.
     */
    size_t batch_size;

    /**
     * @brief Stores the dimensions of the input data expected by this layer.
     */
    Dimensions input_dims;

    /**
     * @brief Stores the dimensions of the output data produced by this layer.
     */
    Dimensions output_dims;

    /**
     * @brief The type of activation function applied to the layer's output (e.g., ReLU, Sigmoid).
     */
    ActivationType activation_type;

    // --- OpenCL Buffers ---
    /**
     * @brief OpenCL buffer storing the pre-activation values (weighted sum + bias) before applying the activation function.
     */
    cl::Buffer pre_activations;

    /**
     * @brief OpenCL buffer storing the output activations of the layer after applying the activation function.
     */
    cl::Buffer outputs;

    /**
     * @brief OpenCL buffer storing the error deltas (gradients of the loss with respect to pre-activations) for this layer.
     */
    cl::Buffer deltas;

    /**
     * @brief OpenCL buffer storing the weights connecting this layer's inputs to its outputs.
     */
    cl::Buffer weights;

    /**
     * @brief OpenCL buffer storing the biases applied to this layer's outputs.
     */
    cl::Buffer biases;

    /**
     * @brief OpenCL buffer storing the accumulated gradients for the weights during backpropagation.
     */
    cl::Buffer weight_gradients;

    /**
     * @brief OpenCL buffer storing the accumulated gradients for the biases during backpropagation.
     */
    cl::Buffer bias_gradients;

    // --- OpenCL Context, Command Queue, and Program ---
    /**
     * @brief The OpenCL context, providing an environment for OpenCL objects.
     */
    cl::Context context;

    /**
     * @brief The OpenCL command queue, used to enqueue OpenCL commands to the device.
     */
    cl::CommandQueue queue;

    /**
     * @brief The OpenCL program object, containing the compiled OpenCL kernels.
     */
    cl::Program program;

    /**
     * @brief Constructs a new Layer object.
     *
     * Initializes common layer properties and OpenCL related objects from the provided setup.
     * All layers are currently assumed to have an activation function.
     *
     * @param ocl_setup A reference to the OpenCLSetup object, providing access to OpenCL context,
     * device, and command queue, and program.
     * @param input_dims The dimensions of the input data to this layer.
     * @param output_dims The dimensions of the output data from this layer.
     * @param act_type The type of activation function to use for this layer (default: ReLU).
     * @param batch_size The number of samples processed in one forward/backward pass (default: 1).
     */
    Layer(const OpenCLSetup& ocl_setup,
          const Dimensions& input_dims, const Dimensions& output_dims,
          ActivationType act_type = ActivationType::ReLU, size_t batch_size = 1)
        : context(ocl_setup.context), queue(ocl_setup.queue), program(ocl_setup.program),
          input_dims(input_dims), output_dims(output_dims),
          activation_type(act_type), batch_size(batch_size) {}

    /**
     * @brief Virtual destructor for the Layer class.
     *
     * Ensures that destructors of derived classes are called correctly when
     * deleting a Layer pointer. Uses default destructor behavior.
     */
    virtual ~Layer() = default;

    // --- Pure Virtual Methods for Layer Functionality (Must be Overridden by Derived Classes) ---

    /**
     * @brief Pure virtual method to initialize the weights and biases of the layer.
     *
     * Concrete layer implementations must define how their specific weights and biases
     * are initialized (e.g., Xavier, He, or random initialization).
     */
    virtual void initializeWeightsAndBiases() = 0;

    /**
     * @brief Pure virtual method to perform the forward pass computation for the layer.
     *
     * This involves taking an input, performing the layer's specific computation (e.g., matrix multiplication for dense layer),
     * and applying the activation function. The results are stored in `outputs`.
     *
     * @param input_buffer An OpenCL buffer containing the input data from the previous layer or the network input.
     */
    virtual void runForward(const cl::Buffer& input_buffer) = 0;

    /**
     * @brief Pure virtual method to compute the initial error deltas for the output layer.
     *
     * This method is specifically for the last layer of the network, calculating the
     * difference between the network's predictions and the true labels.
     *
     * @param true_labels_buffer An OpenCL buffer containing the true labels for the current batch.
     * @param loss_function_type The type of loss function used to compute the error
     * (default: Mean Squared Error).
     */
    virtual void computeOutputDeltas(const cl::Buffer& true_labels_buffer, const LossFunctionType& loss_function_type = LossFunctionType::MeanSquaredError) = 0;

    /**
     * @brief Pure virtual method to backpropagate error deltas from the next layer to the current layer.
     *
     * This method calculates the deltas for the current layer based on the deltas
     * of the subsequent layer and the weights connecting them.
     *
     * @param next_layer_weights An OpenCL buffer containing the weights of the next layer.
     * @param next_layer_deltas An OpenCL buffer containing the deltas (gradients of loss w.r.t. pre-activations) of the next layer.
     * @param next_layer_output_size The output size (number of neurons/elements) of the next layer.
     */
    virtual void backpropDeltas(const cl::Buffer& next_layer_weights, const cl::Buffer& next_layer_deltas, const size_t next_layer_output_size) = 0;

    /**
     * @brief Pure virtual method to calculate the gradients for the layer's weights.
     *
     * This method computes how much each weight contributes to the total error,
     * based on the input to the current layer and the deltas of the current layer.
     * These gradients are accumulated for later use in optimization (e.g., Adam, SGD).
     *
     * @param inputs_to_current_layer An OpenCL buffer containing the input activations
     * from the previous layer, which are the inputs to this layer.
     */
    virtual void calculateWeightGradients(const cl::Buffer& inputs_to_current_layer) = 0;

    /**
     * @brief Pure virtual method to calculate the gradients for the layer's biases.
     *
     * This method computes how much each bias contributes to the total error,
     * based on the deltas of the current layer.
     */
    virtual void calculateBiasGradients() = 0;

    // --- Virtual Utility Methods (Can be Overridden by Derived Classes) ---

    /**
     * @brief Sets a new batch size for the layer.
     *
     * This might require re-allocating OpenCL buffers if their sizes depend on the batch size.
     *
     * @param new_batch_size The new batch size to set.
     */
    virtual void setBatchSize(size_t new_batch_size) {
        batch_size = new_batch_size;
    }

    /**
     * @brief Calculates and returns the total number of elements (weights) in this layer.
     *
     * @return The size of the weights buffer in terms of float elements.
     */
    virtual size_t getWeightsSize() const {
        return input_dims.getTotalElements() * output_dims.getTotalElements();
    }

    /**
     * @brief Calculates and returns the total number of elements (biases) in this layer.
     *
     * @return The size of the biases buffer in terms of float elements.
     */
    virtual size_t getBiasesSize() const {
        return output_dims.getTotalElements();
    }

    /**
     * @brief Generates a random float value within a specified range.
     *
     * This utility function is commonly used for initializing weights.
     * It uses a static random engine seeded with the current time to ensure
     * different sequences of random numbers across different runs.
     *
     * @param min The minimum value for the random number.
     * @param max The maximum value for the random number.
     * @return A random float value between min and max (inclusive).
     */
    float getRandomWeight(float min, float max) {
        static std::default_random_engine generator(
            static_cast<unsigned int>(
                std::chrono::system_clock::now().time_since_epoch().count()
            )
        );
        std::uniform_real_distribution<float> distribution(min, max);
        return distribution(generator);
    }

    // --- Debugging and Utility Methods ---

    /**
     * @brief Prints basic information about the layer to the console.
     *
     * Includes the layer type, input/output dimensions, batch size, and activation type.
     */
    virtual void print() const {
        std::cout << "Layer: " << typeid(*this).name() << "\n" // Prints the derived class name
                  << "Input Dimensions: " << input_dims.getTotalElements() << "\n"
                  << "Output Dimensions: " << output_dims.getTotalElements() << "\n"
                  << "Batch Size: " << batch_size << "\n"
                  << "Activation Type: " << static_cast<unsigned int>(activation_type) << "\n";
    }

    /**
     * @brief Reads and prints the contents of an OpenCL buffer to the console.
     *
     * This is a utility method primarily for debugging purposes, allowing inspection
     * of data stored on the GPU.
     *
     * @param buffer The OpenCL buffer to read from.
     * @param size The number of float elements to read from the buffer.
     * @param label A string label to identify the buffer in the output (default: "Buffer").
     */
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
};