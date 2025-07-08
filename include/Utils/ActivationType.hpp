#pragma once

/**
 * @enum ActivationType
 * @brief Defines the types of activation functions supported in the neural network.
 *
 * Activation functions introduce non-linearity into the network, allowing it to
 * learn complex patterns and relationships in the data that linear models cannot.
 * Each enumerator is assigned an `unsigned int` value for easy mapping to
 * OpenCL kernel logic or switch statements.
 */
enum class ActivationType : unsigned int {
    /**
     * @brief Linear activation function.
     * Output is directly proportional to the input (f(x) = x).
     * Used when no non-linearity is desired, often in the output layer for regression tasks.
     */
    Linear = 0,

    /**
     * @brief Rectified Linear Unit (ReLU) activation function.
     * Output is x if x > 0, and 0 otherwise (f(x) = max(0, x)).
     * Widely used in hidden layers due to its computational efficiency and
     * ability to mitigate the vanishing gradient problem.
     */
    ReLU = 1,

    /**
     * @brief Sigmoid activation function.
     * Squashes the input to a range between 0 and 1 (f(x) = 1 / (1 + e^-x)).
     * Commonly used in the output layer for binary classification problems,
     * where the output can be interpreted as a probability.
     */
    Sigmoid = 2,

    /**
     * @brief Hyperbolic Tangent (Tanh) activation function.
     * Squashes the input to a range between -1 and 1 (f(x) = (e^x - e^-x) / (e^x + e^-x)).
     * Often used in hidden layers, similar to Sigmoid, but its output is zero-centered,
     * which can sometimes help with training stability.
     */
    Tanh = 3,
};