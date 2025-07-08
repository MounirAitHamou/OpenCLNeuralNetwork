#pragma once

#include "Utils/OpenCLSetup.hpp"

#include <string>
#include <iostream>
#include <vector>

/**
 * @class Optimizer
 * @brief An abstract base class defining the interface for all neural network optimizers.
 *
 * This class provides common properties and pure virtual functions that every
 * concrete optimization algorithm (e.g., SGD, Adam, AdamW) must implement.
 * It manages the learning rate, weight decay, and OpenCL resources necessary
 * for parameter updates on the GPU.
 */
class Optimizer {
public:
    // --- Optimizer Properties ---

    /**
     * @brief The learning rate for the optimizer.
     * This scalar value controls the step size during parameter updates.
     */
    float learning_rate;

    /**
     * @brief The weight decay rate (L2 regularization strength).
     * This value is used to penalize large weights, helping to prevent overfitting.
     */
    float weight_decay_rate;

    /**
     * @brief The OpenCL command queue.
     * Used to enqueue commands (like kernel executions and buffer reads/writes)
     * to the OpenCL device.
     */
    cl::CommandQueue queue;

    /**
     * @brief The OpenCL program object.
     * Contains the compiled OpenCL kernels that the optimizer might use
     * for updating parameters on the GPU.
     */
    cl::Program program;

    // --- Constructor ---

    /**
     * @brief Constructs a new Optimizer object.
     *
     * Initializes the OpenCL command queue and program from the provided setup,
     * and sets the learning rate and weight decay rate.
     *
     * @param ocl_setup A reference to the OpenCLSetup object, providing access
     * to the OpenCL context, command queue, and program.
     * @param learning_rate The initial learning rate for the optimizer (default: 0.01f).
     * @param weight_decay_rate The initial weight decay rate (default: 0.0f, no decay).
     */
    Optimizer(const OpenCLSetup& ocl_setup,
              float learning_rate = 0.01f,
              float weight_decay_rate = 0.0f)
        : queue(ocl_setup.queue), program(ocl_setup.program),
          learning_rate(learning_rate), weight_decay_rate(weight_decay_rate) {}

    // --- Destructor ---

    /**
     * @brief Virtual destructor for the Optimizer class.
     *
     * Ensures that destructors of derived classes are called correctly when
     * deleting an Optimizer pointer. Uses default destructor behavior.
     */
    virtual ~Optimizer() = default;

    // --- Pure Virtual Methods (Must be Overridden by Derived Classes) ---

    /**
     * @brief Pure virtual method to update a set of parameters (weights or biases).
     *
     * Concrete optimizer implementations must define how they update parameters
     * based on their corresponding gradients and internal state (e.g., moments for Adam).
     * This method typically involves launching an OpenCL kernel.
     *
     * @param params_buf An OpenCL buffer containing the parameters to be updated.
     * @param grads_buf An OpenCL buffer containing the gradients for the `params_buf`.
     * @param num_elements The total number of float elements in `params_buf` and `grads_buf`.
     */
    virtual void updateParameters(cl::Buffer& params_buf,
                                  cl::Buffer& grads_buf,
                                  size_t num_elements) = 0;

    /**
     * @brief Virtual method to perform a single optimization step.
     *
     * This method is called once per training iteration (e.g., after all gradients
     * for a batch have been computed and accumulated). Derived classes can override
     * this to perform global updates or increment internal counters (like time step `t` for Adam).
     * The default implementation does nothing.
     */
    virtual void step() {}

    /**
     * @brief Pure virtual method to print the optimizer's configuration.
     *
     * Concrete optimizer implementations must define how they print their
     * specific hyperparameters and state to the console for debugging or logging.
     */
    virtual void print() const = 0;

    // --- Utility Method ---

    /**
     * @brief Reads and prints the contents of an OpenCL buffer to the console.
     *
     * This is a utility method primarily for debugging purposes, allowing inspection
     * of data stored on the GPU. It performs a blocking read operation.
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