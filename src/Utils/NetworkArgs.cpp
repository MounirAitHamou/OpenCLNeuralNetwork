#include "Utils/NetworkArgs.hpp"

namespace Utils {
    NetworkArgs createNetworkArgs(
        const Dimensions& p_initialInputDimensions,
        std::vector<std::unique_ptr<LayerArgs>> p_layerArguments,
        std::unique_ptr<OptimizerArgs> p_optimizerArguments,
        std::unique_ptr<LossFunctionArgs> p_lossFunctionArguments
    ) {
        return NetworkArgs(
            p_initialInputDimensions,
            std::move(p_layerArguments),
            std::move(p_optimizerArguments),
            std::move(p_lossFunctionArguments)
        );
    }
}