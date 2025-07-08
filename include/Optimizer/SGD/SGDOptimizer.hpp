#pragma once

#include "Optimizer/Optimizer.hpp"
#include "Utils/OpenCLSetup.hpp"

#include <iostream>

/**
 * @class SGDOptimizer
 * @brief Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
 *
 * SGD is a fundamental optimization algorithm used to train neural networks.
 * It updates parameters (weights and biases) in the direction opposite to the
 * gradient of the loss function with respect to those parameters, scaled by a learning rate.
 * This implementation also supports optional weight decay (L2 regularization).
 */
class SGDOptimizer : public Optimizer {
public:
    // --- Constructor ---

    /**
     * @brief Constructs a new SGDOptimizer object.
     *
     * Initializes the base Optimizer's properties: learning rate and weight decay rate.
     * SGD is a relatively simple optimizer, so it doesn't have additional hyperparameters
     * like momentum or adaptive learning rates (unlike Adam or AdamW).
     *
     * @param ocl_setup A reference to the OpenCLSetup object, providing access
     * to OpenCL context, device, and command queue.
     * @param learning_rate The global learning rate for the optimizer. This controls
     * how large of a step is taken in the direction of the negative gradient.
     * @param weight_decay_rate The rate for L2 regularization (weight decay). This
     * adds a penalty to the loss function proportional to the square of the weights,
     * encouraging smaller weights and helping to prevent overfitting.
     */
    SGDOptimizer(const OpenCLSetup& ocl_setup,
                 float learning_rate,
                 float weight_decay_rate)
        : Optimizer(ocl_setup, learning_rate, weight_decay_rate) {} // Initialize base class

    // --- Destructor ---

    /**
     * @brief Destroys the SGDOptimizer object.
     * Uses default destructor behavior as there are no dynamically allocated resources
     * specific to SGDOptimizer that are not managed by the base class or OpenCL handles.
     */
    ~SGDOptimizer() = default;

    // --- Virtual Methods (Overrides from Optimizer Base Class) ---

    /**
     * @brief Updates the parameters (weights or biases) using the SGD optimization algorithm.
     *
     * This method applies the core SGD update rule:
     * `param = param - learning_rate * (gradient + weight_decay_rate * param)`
     * The operation is performed on the GPU using OpenCL kernels.
     *
     * @param params_buf An OpenCL buffer containing the parameters (weights or biases) to be updated.
     * @param grads_buf An OpenCL buffer containing the gradients for the corresponding parameters.
     * @param num_elements The total number of float elements in the `params_buf` and `grads_buf`.
     */
    void updateParameters(cl::Buffer& params_buf,
                          cl::Buffer& grads_buf,
                          size_t num_elements) override;

    /**
     * @brief Prints the configuration of the SGD optimizer to the console.
     *
     * Overrides the virtual `print` method from the base `Optimizer` class.
     * It displays the learning rate and weight decay rate specific to this optimizer.
     */
    void print() const override {
        std::cout << "SGD Optimizer:\n"
                  << "Learning Rate: " << learning_rate << "\n"
                  << "Weight Decay Rate: " << weight_decay_rate << "\n";
    }
};