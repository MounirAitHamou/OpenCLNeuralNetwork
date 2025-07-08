#pragma once

#include "Utils/LayerConfig.hpp"
#include "Utils/OptimizerConfig.hpp"
#include "Layer/Layer.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>

/**
 * @class NeuralNetwork
 * @brief Represents a complete neural network, managing a sequence of layers,
 * an optimizer, and the overall training and inference processes.
 *
 * This class orchestrates the forward pass, backpropagation, and weight updates
 * using OpenCL for GPU acceleration.
 */
class NeuralNetwork {
public:
    // --- Neural Network Properties and Components ---

    /**
     * @brief A vector of unique pointers to `Layer` objects, representing the
     * sequence of layers in the neural network. `std::unique_ptr` ensures
     * proper memory management and ownership.
     */
    std::vector<std::unique_ptr<Layer>> layers;

    /**
     * @brief The batch size used for training and inference. This determines
     * how many samples are processed in one forward/backward pass.
     */
    size_t batch_size;

    /**
     * @brief The OpenCL context, providing an environment for OpenCL objects
     * and operations across all layers.
     */
    cl::Context context;

    /**
     * @brief The OpenCL command queue, used to enqueue OpenCL commands to the device
     * for all layers and optimizer operations.
     */
    cl::CommandQueue queue;

    /**
     * @brief The OpenCL program object, containing the compiled OpenCL kernels
     * used by various layers and the optimizer.
     */
    cl::Program program;

    /**
     * @brief A unique pointer to an `Optimizer` object, responsible for updating
     * the network's weights and biases based on calculated gradients.
     */
    std::unique_ptr<Optimizer> optimizer;

    /**
     * @brief The type of loss function used to evaluate the network's performance
     * and compute initial deltas during backpropagation.
     */
    LossFunctionType loss_function_type;

    // --- Constructors ---

    /**
     * @brief Default constructor for NeuralNetwork.
     * Initializes member variables to their default states.
     */
    NeuralNetwork() = default;

    /**
     * @brief Constructs a NeuralNetwork object with specified OpenCL setup and
     * network arguments.
     *
     * This constructor initializes the OpenCL context, queue, program, batch size,
     * and loss function type. It then dynamically creates and configures the
     * layers and the optimizer based on the provided `network_args`.
     *
     * @param ocl_setup A reference to the OpenCLSetup object, providing access
     * to the OpenCL context, device, and command queue.
     * @param network_args A `LayerConfig::NetworkArgs` struct containing
     * overall network configuration, including initial input
     * dimensions, layer arguments, and optimizer parameters.
     */
    NeuralNetwork(const OpenCLSetup& ocl_setup, LayerConfig::NetworkArgs network_args)
        : context(ocl_setup.context), queue(ocl_setup.queue), program(ocl_setup.program),
          batch_size(network_args.batch_size), loss_function_type(network_args.loss_function_type) {
        
        Dimensions current_input_dims = network_args.initial_input_dims;
        // Iterate through the layer arguments to create and add each layer to the network.
        for (const auto& layer_args : network_args.layer_arguments) {
            layer_args->batch_size = batch_size; // Ensure layer's batch size matches network's.
            // Use the factory method to create a concrete Layer object and add it to the vector.
            // The output dimensions of the current layer become the input dimensions for the next.
            layers.emplace_back(layer_args->createLayer(ocl_setup, current_input_dims));
            current_input_dims = layers.back()->output_dims; // Update input dimensions for the next layer.
        }
        // Create the optimizer based on the provided optimizer parameters.
        optimizer = network_args.optimizer_parameters->createOptimizer(ocl_setup);
    }

    // --- Core Neural Network Functionality ---

    /**
     * @brief Performs the forward pass through all layers of the neural network.
     *
     * Takes an initial input batch and propagates it through each layer,
     * computing outputs.
     *
     * @param initial_input_batch An OpenCL buffer containing the input data for the first layer.
     * @param current_batch_actual_size The actual number of samples in the current batch,
     * which might be less than `batch_size` for the last batch.
     * @return An OpenCL buffer containing the final output of the network.
     */
    cl::Buffer forward(const cl::Buffer& initial_input_batch, size_t current_batch_actual_size);

    /**
     * @brief Performs the backpropagation algorithm through all layers of the network.
     *
     * Calculates the error deltas for each layer and computes the gradients
     * for weights and biases.
     *
     * @param original_network_input_batch An OpenCL buffer containing the original
     * input data that was fed into the network
     * during the forward pass. This is needed
     * for calculating gradients in earlier layers.
     * @param target_batch_buf An OpenCL buffer containing the true labels (targets)
     * for the current batch.
     */
    void backprop(const cl::Buffer& original_network_input_batch, const cl::Buffer& target_batch_buf);

    /**
     * @brief Trains the neural network using the provided input and target data.
     *
     * This method orchestrates the entire training loop, including iterating
     * over epochs, batching data, performing forward and backward passes,
     * and updating weights using the optimizer.
     *
     * @param inputs A `std::vector` of `std::vector<float>` representing the
     * entire dataset of input features.
     * @param targets A `std::vector` of `std::vector<float>` representing the
     * corresponding true labels for the input data.
     * @param epochs The number of full passes over the training dataset.
     */
    void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs);

    // --- Utility and Debugging Methods ---

    /**
     * @brief Reads and prints the contents of an OpenCL buffer to the console.
     *
     * This is a utility method primarily for debugging purposes, allowing inspection
     * of data stored on the GPU. It performs a blocking read.
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