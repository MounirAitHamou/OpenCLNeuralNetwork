#pragma once

/**
 * @enum LossFunctionType
 * @brief Defines the types of loss functions supported in the neural network.
 *
 * Loss functions (also known as cost functions or error functions) quantify
 * the difference between the predicted output of the network and the true target values.
 * The goal of training a neural network is to minimize this loss.
 * Each enumerator is assigned an `unsigned int` value for easy mapping to
 * OpenCL kernel logic or switch statements.
 */
enum class LossFunctionType : unsigned int {
    /**
     * @brief Mean Squared Error (MSE) loss function.
     * Calculates the average of the squared differences between predicted and true values.
     * Commonly used for regression tasks, but can also be used for binary classification
     * when the output is a probability.
     * Formula: MSE = (1/n) * Σ(y_pred - y_true)²
     */
    MeanSquaredError = 0,

    /**
     * @brief Binary Cross-Entropy (BCE) loss function.
     * Used for binary classification problems where the output is a probability (between 0 and 1).
     * It measures the performance of a classification model whose output is a probability
     * value between 0 and 1.
     * Formula: BCE = - (1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
     */
    BinaryCrossEntropy = 1,
};