#pragma once

#include "Utils/LayerArgs.hpp"
#include "Utils/OptimizerArgs.hpp"
#include "Utils/LossFunctionArgs.hpp"

namespace Utils {
    struct NetworkArgs {
    private:

        Dimensions m_initialInputDimensions;
        std::vector<std::unique_ptr<LayerArgs>> m_layersArguments;
        std::unique_ptr<OptimizerArgs> m_optimizerArguments;
        std::unique_ptr<LossFunctionArgs> m_lossFunctionArguments;

    public:
        NetworkArgs()
            : m_initialInputDimensions(Dimensions({1})) {}

        NetworkArgs(Dimensions p_initialInputDimensions)
            : m_initialInputDimensions(p_initialInputDimensions) {}

        NetworkArgs(Dimensions p_initialInputDimensions,
                    std::vector<std::unique_ptr<LayerArgs>>&& p_layersArguments,
                    std::unique_ptr<OptimizerArgs>&& p_optimizerArguments,
                    std::unique_ptr<LossFunctionArgs>&& p_lossFunctionArguments)
            :   m_initialInputDimensions(p_initialInputDimensions),
                m_layersArguments(std::move(p_layersArguments)),
                m_optimizerArguments(std::move(p_optimizerArguments)),
                m_lossFunctionArguments(std::move(p_lossFunctionArguments)) {}

        NetworkArgs(Dimensions p_initialInputDimensions,
                    std::vector<std::unique_ptr<LayerArgs>>&& p_layersArguments,
                    std::unique_ptr<LossFunctionArgs>&& p_lossFunctionArguments)
            :   m_initialInputDimensions(p_initialInputDimensions),
                m_layersArguments(std::move(p_layersArguments)), 
                m_lossFunctionArguments(std::move(p_lossFunctionArguments)) {}

        const Dimensions& getInitialInputDimensions() const {
            return m_initialInputDimensions;
        }

        const std::vector<std::unique_ptr<LayerArgs>>& getLayersArguments() const {
            return m_layersArguments;
        }

        const std::unique_ptr<OptimizerArgs>& getOptimizerArguments() const {
            return m_optimizerArguments;
        }

        const std::unique_ptr<LossFunctionArgs>& getLossFunctionArguments() const {
            return m_lossFunctionArguments;
        }
    };

    NetworkArgs createNetworkArgs(
        const Dimensions& p_initialInputDimensions,
        std::vector<std::unique_ptr<LayerArgs>> p_layerArguments,
        std::unique_ptr<OptimizerArgs> p_optimizerArguments,
        std::unique_ptr<LossFunctionArgs> p_lossFunctionArguments
    );
}