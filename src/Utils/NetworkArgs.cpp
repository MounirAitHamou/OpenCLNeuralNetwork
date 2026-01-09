#include "Utils/NetworkArgs.hpp"

namespace Utils {
    NetworkArgs createNetworkArgs(
        const Dimensions& p_initialInputDimensions,
        std::vector<std::unique_ptr<LayerArgs>> p_layerArguments,
        std::unique_ptr<OptimizerArgs> p_optimizerArguments,
        LossFunctionType p_lossFunctionType
    ) {
        return NetworkArgs(
            p_initialInputDimensions,
            std::move(p_layerArguments),
            std::move(p_optimizerArguments),
            p_lossFunctionType
        );
    }
}