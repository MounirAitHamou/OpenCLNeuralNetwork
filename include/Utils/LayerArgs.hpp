#pragma once

#include "Layers/AllLayers.hpp"
#include "Utils/LossFunctionType.hpp"

#include <memory>
#include <stdexcept>
#include <vector>
#include <iterator>

namespace Utils {
   struct LayerArgs {
        virtual ~LayerArgs() = default;
        virtual LayerType getLayerType() const = 0;
        virtual std::unique_ptr<Layer> createLayer(
            const size_t p_layerId,
            std::shared_ptr<Utils::SharedResources> p_sharedResources,
            const Dimensions& p_inputDimensions,
            const size_t p_batchSize,
            std::mt19937& p_rng
        ) const = 0;
    };

    struct DenseLayerArgs : public LayerArgs {
    private:
        Dimensions m_outputDimensions;

    public:
        DenseLayerArgs(Dimensions p_outputDimensions)
            : m_outputDimensions(p_outputDimensions) {}

        std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, const Dimensions& p_inputDimensions, const size_t p_batchSize, std::mt19937& p_rng) const final override {
            return std::make_unique<DenseLayer>(p_layerId, p_sharedResources, p_inputDimensions, m_outputDimensions, p_batchSize, p_rng);
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
        ConvolutionalLayerArgs(FilterDimensions p_filterDimensions, StrideDimensions p_strideDimensions, PaddingType p_paddingType)
            : m_filterDimensions(p_filterDimensions),
              m_strideDimensions(p_strideDimensions),
              m_paddingType(p_paddingType){}

        std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, const Dimensions& p_inputDimensions, const size_t p_batchSize, std::mt19937& p_rng) const final override {
            if (p_inputDimensions.getDimensions()[0] != m_filterDimensions.getInputChannels()) {
                std::cerr << "Error: Input channels of filter dimensions (" << m_filterDimensions.getInputChannels() 
                          << ") do not match the channels of input dimensions (" << p_inputDimensions.getDimensions()[0] << ")." << std::endl;
                throw std::invalid_argument("Input dimensions' channels do not match filter's input channels.");
            }
            return std::make_unique<ConvolutionalLayer>(p_layerId, p_sharedResources, p_inputDimensions, m_filterDimensions, m_strideDimensions, m_paddingType, p_batchSize, p_rng);
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

    struct LeakyReLULayerArgs : public LayerArgs {
    private:
        float m_alpha;
    public:
        LeakyReLULayerArgs(float p_alpha) : m_alpha(p_alpha) {}

        std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, const Dimensions& p_inputDimensions, const size_t p_batchSize, std::mt19937& p_rng) const final override {
            return std::make_unique<LeakyReLULayer>(p_layerId, p_sharedResources, p_inputDimensions, m_alpha, p_batchSize);
        }
        LayerType getLayerType() const override {
            return LayerType::LeakyReLU;
        }
    };

    struct ReLULayerArgs : public LayerArgs {
    public:
        ReLULayerArgs() = default;

        std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, const Dimensions& p_inputDimensions, const size_t p_batchSize, std::mt19937& p_rng) const final override {
            return std::make_unique<ReLULayer>(p_layerId, p_sharedResources, p_inputDimensions, p_batchSize);
        }
        LayerType getLayerType() const override {
            return LayerType::ReLU;
        }
    };

    struct SigmoidLayerArgs : public LayerArgs {
    public:
        SigmoidLayerArgs() = default;

        std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, const Dimensions& p_inputDimensions, const size_t p_batchSize, std::mt19937& p_rng) const final override {
            return std::make_unique<SigmoidLayer>(p_layerId, p_sharedResources, p_inputDimensions, p_batchSize);
        }
        LayerType getLayerType() const override {
            return LayerType::Sigmoid;
        }
    };

    struct SoftmaxLayerArgs : public LayerArgs {
    public:
        SoftmaxLayerArgs() = default;

        std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, const Dimensions& p_inputDimensions, const size_t p_batchSize, std::mt19937& p_rng) const final override {
            return std::make_unique<SoftmaxLayer>(p_layerId, p_sharedResources, p_inputDimensions, p_batchSize);
        }

        LayerType getLayerType() const override {
            return LayerType::Softmax;
        }
    };

    struct TanhLayerArgs : public LayerArgs {
    public:
        TanhLayerArgs() = default;

        std::unique_ptr<Layer> createLayer(const size_t p_layerId, std::shared_ptr<Utils::SharedResources> p_sharedResources, const Dimensions& p_inputDimensions, const size_t p_batchSize, std::mt19937& p_rng) const final override {
            return std::make_unique<TanhLayer>(p_layerId, p_sharedResources, p_inputDimensions, p_batchSize);
        }

        LayerType getLayerType() const override {
            return LayerType::Tanh;
        }
    };


    
    std::unique_ptr<LayerArgs> makeDenseLayerArgs(
        const Dimensions& p_outputDimensions
    );

    std::unique_ptr<LayerArgs> makeConvolutionalLayerArgs(
        const FilterDimensions& p_filterDimensions,
        const StrideDimensions& p_strideDimensions,
        PaddingType p_paddingType
    );

    std::unique_ptr<LayerArgs> makeReLULayerArgs();

    std::unique_ptr<LayerArgs> makeLeakyReLULayerArgs(
        float p_alpha
    );

    std::unique_ptr<LayerArgs> makeSigmoidLayerArgs();

    std::unique_ptr<LayerArgs> makeSoftmaxLayerArgs();

    std::unique_ptr<LayerArgs> makeTanhLayerArgs();

    std::unique_ptr<Layer> loadLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources, const H5::Group& p_layerGroup, const size_t p_batchSize);
}