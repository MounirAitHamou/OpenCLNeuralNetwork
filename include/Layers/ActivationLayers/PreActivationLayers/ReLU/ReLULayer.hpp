#pragma once

#include "Layers/ActivationLayers/PreActivationLayers/PreActivationLayer.hpp"
namespace Layers::Activation {
    class ReLULayer : public PreActivationLayer {
    public:

        ReLULayer(const size_t p_layerId, 
                std::shared_ptr<Utils::SharedResources> p_sharedResources,
                const Utils::Dimensions& p_outputDimensions, 
                const size_t p_batchSize)
            : PreActivationLayer(p_layerId, p_sharedResources, p_outputDimensions, p_batchSize) 
        {
            setupKernels();
        }

        ReLULayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                const H5::Group& p_layerGroup,
                const size_t p_batchSize)
            : PreActivationLayer(p_sharedResources, p_layerGroup, p_batchSize)
        {
            setupKernels();
        }

        ~ReLULayer() = default;

        Utils::LayerType getType() const final override { return Utils::LayerType::ReLU; }
    private:
        void setupKernels() final override;

        void saveReLULayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const { saveLayer(p_layerGroup); }
        bool reLULayerEquals(const cl::CommandQueue& p_queue, const ReLULayer& p_other) const { return layerEquals(p_queue, p_other); }
        void printReLULayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const { printPreActivationLayer(p_queue, p_batchSize); }
    };
}
