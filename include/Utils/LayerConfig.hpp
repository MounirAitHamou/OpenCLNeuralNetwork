#pragma once

#include "Layer/AllLayers.hpp"
#include "Utils/OptimizerConfig.hpp"
#include "Utils/LossFunctionType.hpp"

#include <memory>
#include <stdexcept>
#include <vector>
#include <iterator>

/**
 * @namespace LayerConfig
 * @brief Provides structures and factory functions for configuring neural network layers and the overall network architecture.
 *
 * This namespace separates the configuration logic from the core neural network
 * and layer implementations, promoting a cleaner design and easier network definition.
 */
namespace LayerConfig {

    /**
     * @enum LayerType
     * @brief Enumerates the different types of concrete layers that can be configured.
     * This allows for a clear distinction and type-checking when creating layers.
     */
    enum class LayerType {
        /**
         * @brief Represents a fully connected (dense) layer.
         */
        Dense,
        // Add other layer types here as they are implemented, e.g.:
        // Convolutional,
        // Pooling,
        // Dropout,
    };

    /**
     * @struct LayerArgs
     * @brief An abstract base structure for arguments required to construct any neural network layer.
     *
     * Derived structs will provide specific arguments for their respective layer types.
     * This acts as a factory pattern base for creating `Layer` objects.
     */
    struct LayerArgs {
        /**
         * @brief The desired output dimensions of this layer.
         */
        Dimensions output_dims;

        /**
         * @brief The activation function to be used by this layer.
         */
        ActivationType activation_type;

        /**
         * @brief The batch size for which this layer is configured.
         * This value will be set by the `NetworkArgs` or `NeuralNetwork` class.
         */
        size_t batch_size;

        /**
         * @brief Virtual destructor to ensure proper cleanup of derived `LayerArgs` objects.
         */
        virtual ~LayerArgs() = default;

        /**
         * @brief Constructs a LayerArgs object with common layer parameters.
         *
         * @param out_dims The output dimensions of the layer.
         * @param activation_type The activation function type for the layer.
         * @param batch_size The batch size the layer will operate on.
         */
        LayerArgs(Dimensions out_dims, ActivationType activation_type, size_t batch_size)
            : output_dims(out_dims), activation_type(activation_type), batch_size(batch_size) {}

        /**
         * @brief Pure virtual method to create a concrete `Layer` object.
         *
         * Derived `LayerArgs` structs must implement this method to instantiate
         * their specific `Layer` type.
         *
         * @param ocl_setup The OpenCL setup (context, queue, program) for the layer.
         * @param in_dims The input dimensions to this layer, determined by the previous layer's output.
         * @return A `std::unique_ptr` to the newly created `Layer` object.
         */
        virtual std::unique_ptr<Layer> createLayer(const OpenCLSetup& ocl_setup, Dimensions in_dims) const = 0;

        /**
         * @brief Pure virtual method to get the type of the layer.
         *
         * Derived `LayerArgs` structs must implement this to return their specific `LayerType`.
         *
         * @return The `LayerType` enumeration value for the concrete layer.
         */
        virtual LayerType getLayerType() const = 0;
    };

    /**
     * @struct DenseLayerArgs
     * @brief Concrete implementation of `LayerArgs` for configuring a `DenseLayer`.
     *
     * This struct holds the necessary parameters to construct a `DenseLayer`
     * and provides the factory method to create it.
     */
    struct DenseLayerArgs : public LayerArgs {
        /**
         * @brief Constructs a DenseLayerArgs object.
         *
         * @param out_dims The output dimensions of the dense layer.
         * @param activation_type The activation function type for the dense layer.
         * @param batch_size The batch size the dense layer will operate on.
         */
        DenseLayerArgs(Dimensions out_dims, ActivationType activation_type, size_t batch_size)
            : LayerArgs(out_dims, activation_type, batch_size) {}

        /**
         * @brief Creates a `DenseLayer` object based on the stored arguments.
         *
         * @param ocl_setup The OpenCL setup (context, queue, program) for the layer.
         * @param input_dims The input dimensions to this dense layer.
         * @return A `std::unique_ptr` to the newly created `DenseLayer` object.
         */
        std::unique_ptr<Layer> createLayer(const OpenCLSetup& ocl_setup, Dimensions input_dims) const override {
            return std::make_unique<DenseLayer>(ocl_setup, input_dims, output_dims, activation_type, batch_size);
        }

        /**
         * @brief Returns the `LayerType` for a dense layer.
         *
         * @return `LayerType::Dense`.
         */
        LayerType getLayerType() const override {
            return LayerType::Dense;
        }
    };

    /**
     * @struct NetworkArgs
     * @brief A structure to hold the complete configuration for a neural network.
     *
     * This includes the initial input dimensions, a sequence of layer arguments,
     * optimizer parameters, batch size, and the chosen loss function.
     */
    struct NetworkArgs {
        /**
         * @brief The dimensions of the input data that will be fed into the first layer of the network.
         */
        Dimensions initial_input_dims;

        /**
         * @brief A vector of unique pointers to `LayerArgs` objects, defining the
         * sequence and configuration of each layer in the network.
         */
        std::vector<std::unique_ptr<LayerArgs>> layer_arguments;

        /**
         * @brief The global batch size for the entire network's training and inference.
         */
        size_t batch_size;

        /**
         * @brief A unique pointer to an `OptimizerParameters` object, defining
         * the type and hyperparameters of the optimizer to be used for training.
         */
        std::unique_ptr<OptimizerConfig::OptimizerParameters> optimizer_parameters;

        /**
         * @brief The type of loss function to be used for training the network.
         * Defaults to Mean Squared Error.
         */
        LossFunctionType loss_function_type;

        /**
         * @brief Default constructor for NetworkArgs.
         * Initializes with default input dimensions, batch size, and loss function.
         */
        NetworkArgs()
            : initial_input_dims(Dimensions({1})), batch_size(1), loss_function_type(LossFunctionType::MeanSquaredError) {}

        /**
         * @brief Constructor for NetworkArgs with initial input dimensions, batch size, and loss function.
         *
         * @param initial_input_dims The dimensions of the input data to the network.
         * @param batch_size The batch size for training (default: 1).
         * @param loss_function_type The loss function to use (default: MeanSquaredError).
         */
        NetworkArgs(Dimensions initial_input_dims, size_t batch_size = 1,
                    LossFunctionType loss_function_type = LossFunctionType::MeanSquaredError)
            : initial_input_dims(initial_input_dims), batch_size(batch_size),
              loss_function_type(loss_function_type) {}

        /**
         * @brief Full constructor for NetworkArgs, allowing specification of all network parameters.
         *
         * Uses `std::move` for `layers` and `optimizer_params` to efficiently transfer ownership
         * of the `unique_ptr`s.
         *
         * @param initial_input_dims The dimensions of the input data to the network.
         * @param layers A `std::vector` of `std::unique_ptr<LayerArgs>` representing the layers.
         * @param optimizer_params A `std::unique_ptr<OptimizerConfig::OptimizerParameters>` for the optimizer.
         * @param batch_size The batch size for training (default: 1).
         * @param loss_function_type The loss function to use (default: MeanSquaredError).
         */
        NetworkArgs(Dimensions initial_input_dims,
                    std::vector<std::unique_ptr<LayerArgs>>&& layers,
                    std::unique_ptr<OptimizerConfig::OptimizerParameters>&& optimizer_params,
                    size_t batch_size = 1,
                    LossFunctionType loss_function_type = LossFunctionType::MeanSquaredError)
            : initial_input_dims(initial_input_dims),
              layer_arguments(std::move(layers)), // Efficiently moves ownership of layer arguments
              batch_size(batch_size),
              optimizer_parameters(std::move(optimizer_params)), // Efficiently moves ownership of optimizer parameters
              loss_function_type(loss_function_type) {}
    };

    /**
     * @brief Factory function to create a `NetworkArgs` object.
     *
     * This function simplifies the creation of a `NetworkArgs` object by
     * taking individual layer configurations and combining them. It handles
     * moving the unique pointers into the `NetworkArgs` structure.
     *
     * @param initial_input_dims The dimensions of the input data to the network.
     * @param hidden_layer_configs A vector of unique pointers to `LayerArgs` for hidden layers.
     * @param output_layer_config A unique pointer to `LayerArgs` for the output layer.
     * @param optimizer_parameters A reference to `OptimizerConfig::OptimizerParameters` for the optimizer.
     * @param batch_size The batch size for training (default: 1).
     * @param loss_function_type The loss function to use (default: MeanSquaredError).
     * @return A `NetworkArgs` object configured with the specified parameters.
     */
    NetworkArgs createNetworkArgs(
        const Dimensions& initial_input_dims,
        std::vector<std::unique_ptr<LayerArgs>> hidden_layer_configs,
        std::unique_ptr<LayerArgs> output_layer_config,
        const OptimizerConfig::OptimizerParameters& optimizer_parameters,
        size_t batch_size = 1,
        LossFunctionType loss_function_type = LossFunctionType::MeanSquaredError
    );

    /**
     * @brief Factory function to create a `DenseLayerArgs` object.
     *
     * This function simplifies the creation of arguments for a dense layer.
     *
     * @param layer_dimensions The output dimensions for the dense layer.
     * @param activation_type The activation function for the dense layer (default: ReLU).
     * @param batch_size The batch size for the dense layer (default: 1).
     * @return A `std::unique_ptr` to a newly created `DenseLayerArgs` object.
     */
    std::unique_ptr<DenseLayerArgs> makeDenseLayerArgs(
        const Dimensions& layer_dimensions,
        ActivationType activation_type = ActivationType::ReLU,
        size_t batch_size = 1
    );
} // namespace LayerConfig