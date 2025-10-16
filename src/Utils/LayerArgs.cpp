#include "Utils/LayerArgs.hpp"

namespace Utils {
    std::unique_ptr<DenseLayerArgs> makeDenseLayerArgs(
        const Dimensions& p_outputDimensions,
        ActivationType p_activationType
    ) {
        if (p_outputDimensions.getDimensions().size() != 1) {
            throw std::invalid_argument("Dense layer dimensions must be a single value for output size.");
        }
        return std::make_unique<DenseLayerArgs>(
            p_outputDimensions,
            p_activationType
        );
    }

    std::unique_ptr<ConvolutionalLayerArgs> makeConvolutionalLayerArgs(
        const FilterDimensions& p_filterDimensions,
        const StrideDimensions& p_strideDimensions,
        PaddingType p_paddingType,
        ActivationType p_activationType
    ) {
        return std::make_unique<ConvolutionalLayerArgs>(
            p_filterDimensions,
            p_strideDimensions,
            p_paddingType,
            p_activationType
        );
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
            default:
                throw std::runtime_error("Unsupported layer type: " + std::to_string(layerType));
        }
    }
}