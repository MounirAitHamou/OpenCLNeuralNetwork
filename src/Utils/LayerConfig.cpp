#include "Utils/LayerConfig.hpp"

namespace LayerConfig {
    NetworkArgs createNetworkArgs(
        const Dimensions& initial_input_dims,
        std::vector<std::unique_ptr<LayerArgs>> hidden_layer_configs,
        std::unique_ptr<LayerArgs> output_layer_config,
        const OptimizerConfig::OptimizerParameters& optimizer_parameters,
        size_t batch_size,
        LossFunctionType loss_function_type
    ) {
        std::unique_ptr<OptimizerConfig::OptimizerParameters> opt_params_ptr;
        switch (optimizer_parameters.getOptimizerType()) {
            case OptimizerConfig::OptimizerType::SGD: {
                opt_params_ptr = std::make_unique<OptimizerConfig::SGDOptimizerParameters>(
                    optimizer_parameters.learning_rate,
                    optimizer_parameters.weight_decay_rate);
                break;
            }
            case OptimizerConfig::OptimizerType::Adam: {
                const auto& adam_params = dynamic_cast<const OptimizerConfig::AdamOptimizerParameters&>(optimizer_parameters);
                opt_params_ptr = std::make_unique<OptimizerConfig::AdamOptimizerParameters>(
                    adam_params.learning_rate,
                    adam_params.weight_decay_rate,
                    adam_params.beta1,
                    adam_params.beta2,
                    adam_params.epsilon);
                break;
            }
            case OptimizerConfig::OptimizerType::AdamW: {
                const auto& adamw_params = dynamic_cast<const OptimizerConfig::AdamWOptimizerParameters&>(optimizer_parameters);
                opt_params_ptr = std::make_unique<OptimizerConfig::AdamWOptimizerParameters>(
                    adamw_params.learning_rate,
                    adamw_params.weight_decay_rate,
                    adamw_params.beta1,
                    adamw_params.beta2,
                    adamw_params.epsilon);
                break;
            }
            default:
                throw std::invalid_argument("Unsupported optimizer type provided to createNetworkArgs.");
        }

        std::vector<std::unique_ptr<LayerArgs>> all_layers;
        all_layers.insert(all_layers.end(),
                          std::make_move_iterator(hidden_layer_configs.begin()),
                          std::make_move_iterator(hidden_layer_configs.end()));
        all_layers.push_back(std::move(output_layer_config));

        return NetworkArgs(initial_input_dims, std::move(all_layers), std::move(opt_params_ptr), batch_size, loss_function_type);
    }

    std::unique_ptr<DenseLayerArgs> makeDenseLayerArgs(
        const Dimensions& layer_dimensions,
        ActivationType activation_type,
        size_t batch_size
    ) {
        return std::make_unique<DenseLayerArgs>(
            layer_dimensions,
            activation_type,
            batch_size
        );
    }
}