#include "Utils/LayerConfig.hpp"

namespace LayerConfig {
    
    std::unique_ptr<Layer> loadLayer(const OpenCLSetup& ocl_setup, const H5::Group& layer_group, const size_t batch_size) {
        H5::Attribute type_attr = layer_group.openAttribute("layer_type");
        unsigned int layer_type;
        type_attr.read(H5::PredType::NATIVE_UINT, &layer_type);

        switch (layerTypeFromUint(layer_type)) {
            case LayerType::Dense:
                return std::make_unique<DenseLayer>(ocl_setup, layer_group, batch_size);
            // Add other cases here...
            default:
                throw std::runtime_error("Unsupported layer type: " + std::to_string(layer_type));
        }
    }
    
    std::unique_ptr<DenseLayerArgs> makeDenseLayerArgs(
        const Dimensions& layer_dimensions,
        ActivationType activation_type,
        size_t batch_size
    ) {
        return std::make_unique<DenseLayerArgs>(
            layer_dimensions,
            activation_type,
            batch_size
        );
    }
}