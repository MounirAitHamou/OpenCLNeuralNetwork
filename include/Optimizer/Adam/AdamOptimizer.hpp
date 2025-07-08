#pragma once

#include "Optimizer/Optimizer.hpp"
#include "Utils/OpenCLSetup.hpp"

#include <map>
#include <string>
#include <iostream>

/**
 * @class AdamOptimizer
 * @brief Implements the Adam (Adaptive Moment Estimation) optimization algorithm.
 *
 * Adam is an adaptive learning rate optimization algorithm that computes
 * individual adaptive learning rates for different parameters from estimates
 * of first and second moments of the gradients. It combines ideas from
 * RMSprop and Momentum.
 */
class AdamOptimizer : public Optimizer {
public:
    // --- Adam Specific Hyperparameters ---

    /**
     * @brief Exponential decay rate for the first moment estimates (mean of gradients).
     * Typically set close to 1.0 (e.g., 0.9).
     */
    float beta1;

    /**
     * @brief Exponential decay rate for the second moment estimates (uncentered variance of gradients).
     * Typically set close to 1.0 (e.g., 0.999).
     */
    float beta2;

    /**
     * @brief A small constant for numerical stability to prevent division by zero
     * when dividing by the square root of the second moment estimate.
     */
    float epsilon;

    /**
     * @brief Time step (iteration count) for bias correction.
     * Incremented with each parameter update step.
     */
    unsigned int t;

    // --- OpenCL Buffers for Moments ---

    /**
     * @brief A map to store OpenCL buffers for the first moment (m) and second moment (v)
     * estimates for each parameter buffer (weights and biases) in the network.
     * The key is a pointer to the parameter buffer, and the value is a pair of
     * OpenCL buffers: `first_moment_buffer` and `second_moment_buffer`.
     */
    std::map<cl::Buffer*, std::pair<cl::Buffer, cl::Buffer>> moment_buffers;

    /**
     * @brief The OpenCL context, inherited from the base Optimizer and stored here
     * for direct access when creating new buffers for moments.
     */
    cl::Context context;

    // --- Constructor ---

    /**
     * @brief Constructs a new AdamOptimizer object.
     *
     * Initializes the base Optimizer's properties (learning rate, weight decay rate)
     * and sets Adam-specific hyperparameters.
     *
     * @param ocl_setup A reference to the OpenCLSetup object, providing access
     * to OpenCL context, device, and command queue.
     * @param learning_rate The global learning rate for the optimizer.
     * @param weight_decay_rate The rate for L2 regularization (weight decay).
     * @param beta1 The exponential decay rate for the first moment estimates (default: 0.9f).
     * @param beta2 The exponential decay rate for the second moment estimates (default: 0.999f).
     * @param epsilon A small constant for numerical stability (default: 1e-8f).
     */
    AdamOptimizer(const OpenCLSetup& ocl_setup,
                  float learning_rate,
                  float weight_decay_rate,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float epsilon = 1e-8f)
        : Optimizer(ocl_setup, learning_rate, weight_decay_rate), // Initialize base class
          context(ocl_setup.context), // Store OpenCL context for buffer creation
          beta1(beta1), beta2(beta2), epsilon(epsilon), t(1) {} // Initialize Adam-specific params and time step

    // --- Destructor ---

    /**
     * @brief Destroys the AdamOptimizer object.
     * Uses default destructor behavior as OpenCL buffers are managed by `cl::Buffer`
     * and `std::map` handles its own memory.
     */
    ~AdamOptimizer() = default;

    // --- Virtual Methods (Overrides from Optimizer Base Class) ---

    /**
     * @brief Updates the parameters (weights or biases) using the Adam optimization algorithm.
     *
     * This method calculates and applies the parameter updates based on the
     * gradients, first moment estimates, second moment estimates, and bias corrections.
     * It also incorporates weight decay.
     *
     * @param params_buf An OpenCL buffer containing the parameters (weights or biases) to be updated.
     * @param grads_buf An OpenCL buffer containing the gradients for the corresponding parameters.
     * @param num_elements The total number of float elements in the `params_buf` and `grads_buf`.
     */
    void updateParameters(cl::Buffer& params_buf,
                          cl::Buffer& grads_buf,
                          size_t num_elements) override;

    /**
     * @brief Performs a single optimization step.
     *
     * This method is typically called once per training iteration (after all
     * gradients for a batch have been accumulated). It increments the time step `t`
     * and triggers the update of all registered parameters.
     */
    void step() override;

    /**
     * @brief Prints the configuration and current state of the Adam optimizer to the console.
     *
     * Overrides the virtual `print` method from the base `Optimizer` class.
     */
    void print() const override {
        std::cout << "Adam Optimizer:\n"
                  << "Learning Rate: " << learning_rate << "\n"
                  << "Weight Decay Rate: " << weight_decay_rate << "\n"
                  << "Beta1: " << beta1 << "\n"
                  << "Beta2: " << beta2 << "\n"
                  << "Epsilon: " << epsilon << "\n";
    }
};