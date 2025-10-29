#include "Utils/LayerArgs.hpp"

namespace Utils {
    std::unique_ptr<LayerArgs> makeDenseLayerArgs(
        const Dimensions& p_outputDimensions
    ) {
        if (p_outputDimensions.getDimensions().size() != 1) {
            throw std::invalid_argument("Dense layer dimensions must be a single value for output size.");
        }
        return std::make_unique<DenseLayerArgs>(p_outputDimensions);
    }

    std::unique_ptr<LayerArgs> makeConvolutionalLayerArgs(
        const FilterDimensions& p_filterDimensions,
        const StrideDimensions& p_strideDimensions,
        PaddingType p_paddingType
    ) {
        return std::make_unique<ConvolutionalLayerArgs>(
            p_filterDimensions,
            p_strideDimensions,
            p_paddingType
        );
    }

    std::unique_ptr<LayerArgs> makeReLULayerArgs() {
        return std::make_unique<ReLULayerArgs>();
    }

    std::unique_ptr<LayerArgs> makeLeakyReLULayerArgs(
        float p_alpha
    ) {
        return std::make_unique<LeakyReLULayerArgs>(p_alpha);
    }

    std::unique_ptr<LayerArgs> makeSigmoidLayerArgs() {
        return std::make_unique<SigmoidLayerArgs>();
    }

    std::unique_ptr<LayerArgs> makeSoftmaxLayerArgs() {
        return std::make_unique<SoftmaxLayerArgs>();
    }

    std::unique_ptr<LayerArgs> makeTanhLayerArgs() {
        return std::make_unique<TanhLayerArgs>();
    }

    std::unique_ptr<Layer> loadLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources, 
                                     const H5::Group& p_layerGroup, 
                                     const size_t p_batchSize) {
        unsigned int layerType;
        p_layerGroup.openAttribute("layerType").read(H5::PredType::NATIVE_UINT, &layerType);

        switch (layerTypeFromUint(layerType)) {
            case LayerType::Dense:
                return std::make_unique<DenseLayer>(p_sharedResources, p_layerGroup, p_batchSize);
            case LayerType::Convolutional:
                return std::make_unique<ConvolutionalLayer>(p_sharedResources, p_layerGroup, p_batchSize);
            case LayerType::ReLU:
                return std::make_unique<ReLULayer>(p_sharedResources, p_layerGroup, p_batchSize);
            case LayerType::LeakyReLU:
                return std::make_unique<LeakyReLULayer>(p_sharedResources, p_layerGroup, p_batchSize);
            case LayerType::Sigmoid:
                return std::make_unique<SigmoidLayer>(p_sharedResources, p_layerGroup, p_batchSize);
            case LayerType::Softmax:
                return std::make_unique<SoftmaxLayer>(p_sharedResources, p_layerGroup, p_batchSize);
            case LayerType::Tanh:
                return std::make_unique<TanhLayer>(p_sharedResources, p_layerGroup, p_batchSize);
            default:
                throw std::runtime_error("Unsupported layer type: " + std::to_string(layerType));
        }
    }
}