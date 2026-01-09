#pragma once

#include "Layers/ActivationLayers/ActivationLayer.hpp"
namespace Layers::Activation {
    class SigmoidLayer : public ActivationLayer {
    public:
        SigmoidLayer(const size_t p_layerId, 
                    std::shared_ptr<Utils::SharedResources> p_sharedResources,
                    const Utils::Dimensions& p_outputDimensions, 
                    const size_t p_batchSize)
            : ActivationLayer(p_layerId, p_sharedResources, p_outputDimensions, p_batchSize) 
        {
            setupKernels();
        }

        SigmoidLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                    const H5::Group& p_layerGroup,
                    const size_t p_batchSize)
            : ActivationLayer(p_sharedResources, p_layerGroup, p_batchSize)
        {
            setupKernels();
        }

        ~SigmoidLayer() = default;

        Utils::LayerType getType() const final override { return Utils::LayerType::Sigmoid; }
    private:
        void setupKernels() final override;

        void saveSigmoidLayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const { saveLayer(p_layerGroup); }
        bool sigmoidLayerEquals(const cl::CommandQueue& p_queue, const SigmoidLayer& p_other) const { return layerEquals(p_queue, p_other); }
        void printSigmoidLayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const { printLayer(p_queue, p_batchSize); }
    };
}