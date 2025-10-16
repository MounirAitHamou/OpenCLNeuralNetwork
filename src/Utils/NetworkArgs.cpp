#include "Utils/NetworkArgs.hpp"

namespace Utils {
    NetworkArgs createNetworkArgs(
        const Dimensions& p_initialInputDimensions,
        std::vector<std::unique_ptr<LayerArgs>> p_hiddenLayerArguments,
        std::unique_ptr<LayerArgs> p_outputLayerArguments,
        const OptimizerArgs& p_optimizerArguments,
        size_t p_batchSize,
        LossFunctionType p_lossFunctionType
    ) {
        std::unique_ptr<OptimizerArgs> optArgsPtr;
        switch (p_optimizerArguments.getOptimizerType()) {
            case OptimizerType::SGD: {
                optArgsPtr = std::make_unique<SGDOptimizerArgs>(
                    p_optimizerArguments.getLearningRate(),
                    p_optimizerArguments.getWeightDecayRate());
                break;
            }
            case OptimizerType::Adam: {
                const auto& adamArguments = dynamic_cast<const AdamOptimizerArgs&>(p_optimizerArguments);
                optArgsPtr = std::make_unique<AdamOptimizerArgs>(
                    adamArguments.getLearningRate(),
                    adamArguments.getWeightDecayRate(),
                    adamArguments.getBeta1(),
                    adamArguments.getBeta2(),
                    adamArguments.getEpsilon());
                break;
            }
            case OptimizerType::AdamW: {
                const auto& adamWArguments = dynamic_cast<const AdamWOptimizerArgs&>(p_optimizerArguments);
                optArgsPtr = std::make_unique<AdamWOptimizerArgs>(
                    adamWArguments.getLearningRate(),
                    adamWArguments.getWeightDecayRate(),
                    adamWArguments.getBeta1(),
                    adamWArguments.getBeta2(),
                    adamWArguments.getEpsilon());
                break;
            }
            default:
                throw std::invalid_argument("Unsupported optimizer type provided to createNetworkArgs.");
        }

        std::vector<std::unique_ptr<LayerArgs>> layersArgs;
        layersArgs.insert(layersArgs.end(),
                          std::make_move_iterator(p_hiddenLayerArguments.begin()),
                          std::make_move_iterator(p_hiddenLayerArguments.end()));
        layersArgs.push_back(std::move(p_outputLayerArguments));

        return NetworkArgs(p_initialInputDimensions, std::move(layersArgs), std::move(optArgsPtr), p_batchSize, p_lossFunctionType);
    }


    NetworkArgs createNetworkArgs(
        const Dimensions& p_initialInputDimensions,
        const OptimizerArgs& p_optimizerArguments,
        size_t p_batchSize,
        LossFunctionType p_lossFunctionType
    ) {
        std::unique_ptr<OptimizerArgs> optArgsPtr;
        switch (p_optimizerArguments.getOptimizerType()) {
            case OptimizerType::SGD: {
                optArgsPtr = std::make_unique<SGDOptimizerArgs>(
                    p_optimizerArguments.getLearningRate(),
                    p_optimizerArguments.getWeightDecayRate());
                break;
            }
            case OptimizerType::Adam: {
                const auto& adamArguments = dynamic_cast<const AdamOptimizerArgs&>(p_optimizerArguments);
                optArgsPtr = std::make_unique<AdamOptimizerArgs>(
                    adamArguments.getLearningRate(),
                    adamArguments.getWeightDecayRate(),
                    adamArguments.getBeta1(),
                    adamArguments.getBeta2(),
                    adamArguments.getEpsilon());
                break;
            }
            case OptimizerType::AdamW: {
                const auto& adamWArguments = dynamic_cast<const AdamWOptimizerArgs&>(p_optimizerArguments);
                optArgsPtr = std::make_unique<AdamWOptimizerArgs>(
                    adamWArguments.getLearningRate(),
                    adamWArguments.getWeightDecayRate(),
                    adamWArguments.getBeta1(),
                    adamWArguments.getBeta2(),
                    adamWArguments.getEpsilon());
                break;
            }
            default:
                throw std::invalid_argument("Unsupported optimizer type provided to createNetworkArgs.");
        }

        return NetworkArgs(p_initialInputDimensions, std::vector<std::unique_ptr<LayerArgs>>(), std::move(optArgsPtr), p_batchSize, p_lossFunctionType);
    }
}