#pragma once

#include "Utils/LayerArgs.hpp"
#include "Utils/OptimizerArgs.hpp"

namespace Utils {
    struct NetworkArgs {
    private:

        Dimensions m_initialInputDimensions;

        std::vector<std::unique_ptr<LayerArgs>> m_layersArguments;

        size_t m_batchSize;

        std::unique_ptr<OptimizerArgs> m_optimizerArguments;

        LossFunctionType m_lossFunctionType;

    public:
        NetworkArgs()
            : m_initialInputDimensions(Dimensions({1})), m_batchSize(1), m_lossFunctionType(LossFunctionType::MeanSquaredError) {}

        NetworkArgs(Dimensions p_initialInputDimensions, size_t p_batchSize = 1,
                    LossFunctionType p_lossFunctionType = LossFunctionType::MeanSquaredError)
            : m_initialInputDimensions(p_initialInputDimensions), m_batchSize(p_batchSize),
            m_lossFunctionType(p_lossFunctionType) {}

        NetworkArgs(Dimensions p_initialInputDimensions,
                    std::vector<std::unique_ptr<LayerArgs>>&& p_layersArguments,
                    std::unique_ptr<OptimizerArgs>&& p_optimizerArguments,
                    size_t p_batchSize = 1,
                    LossFunctionType p_lossFunctionType = LossFunctionType::MeanSquaredError)
            :   m_initialInputDimensions(p_initialInputDimensions),
                m_layersArguments(std::move(p_layersArguments)),
                m_optimizerArguments(std::move(p_optimizerArguments)),
                m_batchSize(p_batchSize), m_lossFunctionType(p_lossFunctionType) {}

        const Dimensions& getInitialInputDimensions() const {
            return m_initialInputDimensions;
        }

        const std::vector<std::unique_ptr<LayerArgs>>& getLayersArguments() const {
            return m_layersArguments;
        }

        const size_t getBatchSize() const {
            return m_batchSize;
        }
        
        const std::unique_ptr<OptimizerArgs>& getOptimizerArguments() const {
            return m_optimizerArguments;
        }

        const LossFunctionType getLossFunctionType() const {
            return m_lossFunctionType;
        }
    };

    NetworkArgs createNetworkArgs(
        const Dimensions& p_initialInputDimensions,
        std::vector<std::unique_ptr<LayerArgs>> p_hiddenLayerArguments,
        std::unique_ptr<LayerArgs> p_outputLayerArguments,
        const OptimizerArgs& p_optimizerArguments,
        size_t p_batchSize = 1,
        LossFunctionType p_lossFunctionType = LossFunctionType::MeanSquaredError
    );

    NetworkArgs createNetworkArgs(
        const Dimensions& p_initialInputDimensions,
        const OptimizerArgs& p_optimizerArguments,
        size_t p_batchSize = 1,
        LossFunctionType p_lossFunctionType = LossFunctionType::MeanSquaredError
    );
}