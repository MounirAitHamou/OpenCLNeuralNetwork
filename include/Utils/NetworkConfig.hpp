#pragma once

#include "Utils/LayerConfig.hpp"
#include "Utils/OptimizerConfig.hpp"

namespace NetworkConfig {

    struct NetworkArgs {

        Dimensions initial_input_dims;

        std::vector<std::unique_ptr<LayerConfig::LayerArgs>> layer_arguments;

        size_t batch_size;

        std::unique_ptr<OptimizerConfig::OptimizerParameters> optimizer_parameters;

        LossFunctionType loss_function_type;

        NetworkArgs()
            : initial_input_dims(Dimensions({1})), batch_size(1), loss_function_type(LossFunctionType::MeanSquaredError) {}

        NetworkArgs(Dimensions initial_input_dims, size_t batch_size = 1,
                    LossFunctionType loss_function_type = LossFunctionType::MeanSquaredError)
            : initial_input_dims(initial_input_dims), batch_size(batch_size),
              loss_function_type(loss_function_type) {}

        NetworkArgs(Dimensions initial_input_dims,
                    std::vector<std::unique_ptr<LayerConfig::LayerArgs>>&& layers,
                    std::unique_ptr<OptimizerConfig::OptimizerParameters>&& optimizer_params,
                    size_t batch_size = 1,
                    LossFunctionType loss_function_type = LossFunctionType::MeanSquaredError)
            : initial_input_dims(initial_input_dims),
              layer_arguments(std::move(layers)),
              batch_size(batch_size),
              optimizer_parameters(std::move(optimizer_params)),
              loss_function_type(loss_function_type) {}
    };

    NetworkArgs createNetworkArgs(
        const Dimensions& initial_input_dims,
        std::vector<std::unique_ptr<LayerConfig::LayerArgs>> hidden_layer_configs,
        std::unique_ptr<LayerConfig::LayerArgs> output_layer_config,
        const OptimizerConfig::OptimizerParameters& optimizer_parameters,
        size_t batch_size = 1,
        LossFunctionType loss_function_type = LossFunctionType::MeanSquaredError
    );
}