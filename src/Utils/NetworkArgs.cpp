#include "Utils/NetworkArgs.hpp"

namespace Utils {
    NetworkArgs createNetworkArgs(
        const Dimensions& p_initialInputDimensions,
        std::vector<std::unique_ptr<LayerArgs>> p_layerArguments,
        std::unique_ptr<OptimizerArgs> p_optimizerArguments,
        size_t p_batchSize,
        LossFunctionType p_lossFunctionType
    ) {
        return NetworkArgs(
            p_initialInputDimensions,
            std::move(p_layerArguments),
            std::move(p_optimizerArguments),
            p_batchSize,
            p_lossFunctionType
        );
    }
}