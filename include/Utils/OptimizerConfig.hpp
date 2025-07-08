#pragma once

#include "Optimizer/AllOptimizers.hpp"
#include "Utils/OpenCLSetup.hpp"

#include <memory>
#include <iostream>

/**
 * @namespace OptimizerConfig
 * @brief Provides structures and factory functions for configuring neural network optimizers.
 *
 * This namespace defines the parameters for different optimization algorithms
 * and offers factory methods to create concrete optimizer objects based on these parameters.
 * This approach separates configuration from implementation, making it easier to
 * define and switch between optimizers for network training.
 */
namespace OptimizerConfig {

    /**
     * @enum OptimizerType
     * @brief Enumerates the different types of concrete optimizers that can be configured.
     * This allows for a clear distinction and type-checking when creating optimizers.
     */
    enum class OptimizerType {
        /**
         * @brief Represents the Stochastic Gradient Descent (SGD) optimizer.
         */
        SGD,
        /**
         * @brief Represents the Adam (Adaptive Moment Estimation) optimizer.
         */
        Adam,
        /**
         * @brief Represents the AdamW optimizer, a variant of Adam with decoupled weight decay.
         */
        AdamW,
    };

    /**
     * @struct OptimizerParameters
     * @brief An abstract base structure for arguments required to construct any optimizer.
     *
     * Derived structs will provide specific arguments for their respective optimizer types.
     * This acts as a factory pattern base for creating `Optimizer` objects.
     */
    struct OptimizerParameters {
        /**
         * @brief The learning rate for the optimizer.
         * This controls the step size during parameter updates.
         */
        float learning_rate;

        /**
         * @brief The weight decay rate (L2 regularization strength).
         * This value is used to penalize large weights.
         */
        float weight_decay_rate;

        /**
         * @brief Constructs an OptimizerParameters object with common optimizer parameters.
         *
         * @param lr The learning rate (default: 0.01f).
         * @param wd The weight decay rate (default: 0.0f).
         */
        OptimizerParameters(float lr = 0.01f, float wd = 0.0f)
            : learning_rate(lr), weight_decay_rate(wd) {}

        /**
         * @brief Virtual destructor to ensure proper cleanup of derived `OptimizerParameters` objects.
         */
        virtual ~OptimizerParameters() = default;

        /**
         * @brief Pure virtual method to create a concrete `Optimizer` object.
         *
         * Derived `OptimizerParameters` structs must implement this method to instantiate
         * their specific `Optimizer` type.
         *
         * @param ocl_setup The OpenCL setup (context, queue, program) for the optimizer.
         * @return A `std::unique_ptr` to the newly created `Optimizer` object.
         */
        virtual std::unique_ptr<Optimizer> createOptimizer(const OpenCLSetup& ocl_setup) const = 0;

        /**
         * @brief Pure virtual method to get the type of the optimizer.
         *
         * Derived `OptimizerParameters` structs must implement this to return their specific `OptimizerType`.
         *
         * @return The `OptimizerType` enumeration value for the concrete optimizer.
         */
        virtual OptimizerType getOptimizerType() const = 0;

        /**
         * @brief Pure virtual method to print the optimizer parameters.
         *
         * Derived `OptimizerParameters` structs must implement this to display
         * their specific hyperparameters.
         */
        virtual void print() const = 0;

        /**
         * @brief Virtual clone method to create a deep copy of the OptimizerParameters object.
         * This is crucial when passing `OptimizerParameters` by const reference and needing to
         * store a unique_ptr copy, as seen in `LayerConfig::createNetworkArgs`.
         *
         * @return A `std::unique_ptr` to a new `OptimizerParameters` object that is a copy of `this`.
         */
        virtual std::unique_ptr<OptimizerParameters> clone() const = 0;
    };

    /**
     * @struct SGDOptimizerParameters
     * @brief Concrete implementation of `OptimizerParameters` for configuring an `SGDOptimizer`.
     *
     * This struct holds the necessary parameters to construct an `SGDOptimizer`
     * and provides the factory method to create it.
     */
    struct SGDOptimizerParameters : public OptimizerParameters {
        /**
         * @brief Constructs an SGDOptimizerParameters object.
         *
         * @param lr The learning rate (default: 0.01f).
         * @param wd The weight decay rate (default: 0.0f).
         */
        SGDOptimizerParameters(float lr = 0.01f, float wd = 0.0f)
            : OptimizerParameters(lr, wd) {}

        /**
         * @brief Creates an `SGDOptimizer` object based on the stored parameters.
         *
         * @param ocl_setup The OpenCL setup for the optimizer.
         * @return A `std::unique_ptr` to the newly created `SGDOptimizer` object.
         */
        std::unique_ptr<Optimizer> createOptimizer(const OpenCLSetup& ocl_setup) const override {
            return std::make_unique<SGDOptimizer>(ocl_setup, learning_rate, weight_decay_rate);
        }

        /**
         * @brief Returns the `OptimizerType` for an SGD optimizer.
         *
         * @return `OptimizerType::SGD`.
         */
        OptimizerType getOptimizerType() const override {
            return OptimizerType::SGD;
        }

        /**
         * @brief Prints the parameters specific to the SGD optimizer.
         */
        void print() const override {
            std::cout << "SGD Optimizer Parameters:\n"
                      << "Learning Rate: " << learning_rate << "\n"
                      << "Weight Decay Rate: " << weight_decay_rate << "\n";
        }

        /**
         * @brief Clones this SGDOptimizerParameters object.
         *
         * @return A `std::unique_ptr` to a new `SGDOptimizerParameters` object that is a copy of `this`.
         */
        std::unique_ptr<OptimizerParameters> clone() const override {
            return std::make_unique<SGDOptimizerParameters>(*this);
        }
    };

    /**
     * @struct AdamOptimizerParameters
     * @brief Concrete implementation of `OptimizerParameters` for configuring an `AdamOptimizer`.
     *
     * This struct holds the necessary parameters to construct an `AdamOptimizer`
     * and provides the factory method to create it.
     */
    struct AdamOptimizerParameters : public OptimizerParameters {
        /**
         * @brief Exponential decay rate for the first moment estimates.
         */
        float beta1;
        /**
         * @brief Exponential decay rate for the second moment estimates.
         */
        float beta2;
        /**
         * @brief Small constant for numerical stability.
         */
        float epsilon;

        /**
         * @brief Constructs an AdamOptimizerParameters object.
         *
         * @param lr The learning rate (default: 0.01f).
         * @param wd The weight decay rate (default: 0.0f).
         * @param b1 The beta1 parameter (default: 0.9f).
         * @param b2 The beta2 parameter (default: 0.999f).
         * @param eps The epsilon parameter (default: 1e-8f).
         */
        AdamOptimizerParameters(float lr = 0.01f, float wd = 0.0f,
                                float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
            : OptimizerParameters(lr, wd), beta1(b1), beta2(b2), epsilon(eps) {}

        /**
         * @brief Creates an `AdamOptimizer` object based on the stored parameters.
         *
         * @param ocl_setup The OpenCL setup for the optimizer.
         * @return A `std::unique_ptr` to the newly created `AdamOptimizer` object.
         */
        std::unique_ptr<Optimizer> createOptimizer(const OpenCLSetup& ocl_setup) const override {
            return std::make_unique<AdamOptimizer>(ocl_setup, learning_rate, weight_decay_rate, beta1, beta2, epsilon);
        }

        /**
         * @brief Returns the `OptimizerType` for an Adam optimizer.
         *
         * @return `OptimizerType::Adam`.
         */
        OptimizerType getOptimizerType() const override {
            return OptimizerType::Adam;
        }

        /**
         * @brief Prints the parameters specific to the Adam optimizer.
         */
        void print() const override {
            std::cout << "Adam Optimizer Parameters:\n"
                      << "Learning Rate: " << learning_rate << "\n"
                      << "Weight Decay Rate: " << weight_decay_rate << "\n"
                      << "Beta1: " << beta1 << "\n"
                      << "Beta2: " << beta2 << "\n"
                      << "Epsilon: " << epsilon << "\n";
        }

        /**
         * @brief Clones this AdamOptimizerParameters object.
         *
         * @return A `std::unique_ptr` to a new `AdamOptimizerParameters` object that is a copy of `this`.
         */
        std::unique_ptr<OptimizerParameters> clone() const override {
            return std::make_unique<AdamOptimizerParameters>(*this);
        }
    };

    /**
     * @struct AdamWOptimizerParameters
     * @brief Concrete implementation of `OptimizerParameters` for configuring an `AdamWOptimizer`.
     *
     * This struct holds the necessary parameters to construct an `AdamWOptimizer`
     * and provides the factory method to create it.
     */
    struct AdamWOptimizerParameters : public OptimizerParameters {
        /**
         * @brief Exponential decay rate for the first moment estimates.
         */
        float beta1;
        /**
         * @brief Exponential decay rate for the second moment estimates.
         */
        float beta2;
        /**
         * @brief Small constant for numerical stability.
         */
        float epsilon;

        /**
         * @brief Constructs an AdamWOptimizerParameters object.
         *
         * @param lr The learning rate (default: 0.01f).
         * @param wd The weight decay rate (default: 0.0f).
         * @param b1 The beta1 parameter (default: 0.9f).
         * @param b2 The beta2 parameter (default: 0.999f).
         * @param eps The epsilon parameter (default: 1e-8f).
         */
        AdamWOptimizerParameters(float lr = 0.01f, float wd = 0.0f,
                                 float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
            : OptimizerParameters(lr, wd), beta1(b1), beta2(b2), epsilon(eps) {}

        /**
         * @brief Creates an `AdamWOptimizer` object based on the stored parameters.
         *
         * @param ocl_setup The OpenCL setup for the optimizer.
         * @return A `std::unique_ptr` to the newly created `AdamWOptimizer` object.
         */
        std::unique_ptr<Optimizer> createOptimizer(const OpenCLSetup& ocl_setup) const override {
            return std::make_unique<AdamWOptimizer>(ocl_setup, learning_rate, weight_decay_rate, beta1, beta2, epsilon);
        }

        /**
         * @brief Returns the `OptimizerType` for an AdamW optimizer.
         *
         * @return `OptimizerType::AdamW`.
         */
        OptimizerType getOptimizerType() const override {
            return OptimizerType::AdamW;
        }

        /**
         * @brief Prints the parameters specific to the AdamW optimizer.
         */
        void print() const override {
            std::cout << "AdamW Optimizer Parameters:\n"
                      << "Learning Rate: " << learning_rate << "\n"
                      << "Weight Decay Rate: " << weight_decay_rate << "\n"
                      << "Beta1: " << beta1 << "\n"
                      << "Beta2: " << beta2 << "\n"
                      << "Epsilon: " << epsilon << "\n";
        }

        /**
         * @brief Clones this AdamWOptimizerParameters object.
         *
         * @return A `std::unique_ptr` to a new `AdamWOptimizerParameters` object that is a copy of `this`.
         */
        std::unique_ptr<OptimizerParameters> clone() const override {
            return std::make_unique<AdamWOptimizerParameters>(*this);
        }
    };

    // --- Factory Functions for Optimizer Parameters ---

    /**
     * @brief Factory function to create `SGDOptimizerParameters`.
     *
     * @param learning_rate The learning rate for SGD (default: 0.01f).
     * @param weight_decay_rate The weight decay rate for SGD (default: 0.0f).
     * @return An `SGDOptimizerParameters` object.
     */
    SGDOptimizerParameters makeSGDParameters(float learning_rate = 0.01f, float weight_decay_rate = 0.0f);

    /**
     * @brief Factory function to create `AdamOptimizerParameters`.
     *
     * @param learning_rate The learning rate for Adam (default: 0.01f).
     * @param weight_decay_rate The weight decay rate for Adam (default: 0.0f).
     * @param beta1 The beta1 parameter for Adam (default: 0.9f).
     * @param beta2 The beta2 parameter for Adam (default: 0.999f).
     * @param epsilon The epsilon parameter for Adam (default: 1e-8f).
     * @return An `AdamOptimizerParameters` object.
     */
    AdamOptimizerParameters makeAdamParameters(float learning_rate = 0.01f, float weight_decay_rate = 0.0f,
                                               float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    /**
     * @brief Factory function to create `AdamWOptimizerParameters`.
     *
     * @param learning_rate The learning rate for AdamW (default: 0.01f).
     * @param weight_decay_rate The weight decay rate for AdamW (default: 0.0f).
     * @param beta1 The beta1 parameter for AdamW (default: 0.9f).
     * @param beta2 The beta2 parameter for AdamW (default: 0.999f).
     * @param epsilon The epsilon parameter for AdamW (default: 1e-8f).
     * @return An `AdamWOptimizerParameters` object.
     */
    AdamWOptimizerParameters makeAdamWParameters(float learning_rate = 0.01f, float weight_decay_rate = 0.0f,
                                                 float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    /**
     * @brief Factory function to create a concrete `Optimizer` object based on provided parameters.
     *
     * This function uses the `createOptimizer` virtual method of the `OptimizerParameters`
     * object to instantiate the correct optimizer type.
     *
     * @param ocl_setup The OpenCL setup (context, queue, program) for the optimizer.
     * @param params A constant reference to an `OptimizerParameters` object, which
     * can be any of its derived types (SGD, Adam, AdamW).
     * @return A `std::unique_ptr` to the newly created `Optimizer` object.
     */
    std::unique_ptr<Optimizer> makeOptimizer(const OpenCLSetup& ocl_setup, const OptimizerParameters& params);

} // namespace OptimizerConfig