#pragma once

#include "Layer/AllLayers.hpp"
#include "Utils/LossFunctionType.hpp"

#include <memory>
#include <stdexcept>
#include <vector>
#include <iterator>

namespace Utils {
    struct LayerArgs {
    protected:
        ActivationType m_activationType;

    public:
        virtual ~LayerArgs() = default;

        LayerArgs(ActivationType p_activationType)
            : m_activationType(p_activationType){}

        virtual std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, Dimensions p_inputDimensions, size_t p_batchSize, std::mt19937& p_rng) const = 0;

        virtual LayerType getLayerType() const = 0;

        ActivationType getActivationType() const {
            return m_activationType;
        }
    };

    struct DenseLayerArgs : public LayerArgs {
    private:
        Dimensions m_outputDimensions;

    public:
        DenseLayerArgs(Dimensions p_outputDimensions, ActivationType p_activationType)
            : LayerArgs(p_activationType),
              m_outputDimensions(p_outputDimensions) {}

        std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, Dimensions p_inputDimensions, size_t p_batchSize, std::mt19937& p_rng) const override {
            return std::make_unique<DenseLayer>(p_layerId, p_sharedResources, p_inputDimensions, m_outputDimensions, m_activationType, p_batchSize, p_rng);
        }

        Dimensions getOutputDimensions() const {
            return m_outputDimensions;
        }

        LayerType getLayerType() const override {
            return LayerType::Dense;
        }
    };

    struct ConvolutionalLayerArgs : public LayerArgs {
    private:
        FilterDimensions m_filterDimensions;
        StrideDimensions m_strideDimensions;
        PaddingType m_paddingType;
    public:
        ConvolutionalLayerArgs(FilterDimensions p_filterDimensions, StrideDimensions p_strideDimensions, PaddingType p_paddingType, ActivationType p_activationType)
            : LayerArgs(p_activationType),
              m_filterDimensions(p_filterDimensions),
              m_strideDimensions(p_strideDimensions),
              m_paddingType(p_paddingType){}

        std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, Dimensions p_inputDimensions, size_t p_batchSize, std::mt19937& p_rng) const override {
            if (p_inputDimensions.getDimensions()[0] != m_filterDimensions.getInputChannels()) {
                std::cerr << "Error: Input channels of filter dimensions (" << m_filterDimensions.getInputChannels() 
                          << ") do not match the channels of input dimensions (" << p_inputDimensions.getDimensions()[0] << ")." << std::endl;
                throw std::invalid_argument("Input dimensions' channels do not match filter's input channels.");
            }
            return std::make_unique<ConvolutionalLayer>(p_layerId, p_sharedResources, p_inputDimensions, m_filterDimensions, m_strideDimensions, m_paddingType, m_activationType, p_batchSize, p_rng);
        }

        FilterDimensions getFilterDimensions() const {
            return m_filterDimensions;
        }

        StrideDimensions getStrideDimensions() const {
            return m_strideDimensions;
        }

        PaddingType getPaddingType() const {
            return m_paddingType;
        }

        LayerType getLayerType() const override {
            return LayerType::Convolutional;
        }
    };
    
    std::unique_ptr<DenseLayerArgs> makeDenseLayerArgs(
        const Dimensions& p_outputDimensions,
        ActivationType p_activationType = ActivationType::Linear
    );

    std::unique_ptr<ConvolutionalLayerArgs> makeConvolutionalLayerArgs(
        const FilterDimensions& p_filterDimensions,
        const StrideDimensions& p_strideDimensions,
        PaddingType p_paddingType,
        ActivationType p_activationType = ActivationType::Linear
    );

    std::unique_ptr<Layer> loadLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources, const H5::Group& p_layerGroup, const size_t p_batchSize);
}