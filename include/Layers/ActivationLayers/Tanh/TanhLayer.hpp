#pragma once

#include "Layers/ActivationLayers/ActivationLayer.hpp"

class TanhLayer : public ActivationLayer {
public:

    TanhLayer(const size_t p_layerId, 
              std::shared_ptr<Utils::SharedResources> p_sharedResources,
              const Utils::Dimensions& p_outputDimensions, 
              const size_t p_batchSize)
        : ActivationLayer(p_layerId, p_sharedResources, p_outputDimensions, p_batchSize) 
    {
        setupKernels();
    }

    TanhLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
              const H5::Group& p_layerGroup,
              const size_t p_batchSize)
        : ActivationLayer(p_sharedResources, p_layerGroup, p_batchSize)
    {
        setupKernels();
    }

    ~TanhLayer() = default;

    Utils::LayerType getType() const final override { return Utils::LayerType::Tanh; }

private:
    void setupKernels() final override;

    void saveTanhLayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const { saveLayer(p_layerGroup); }
    bool tanhLayerEquals(const cl::CommandQueue& p_queue, const TanhLayer& p_other) const { return layerEquals(p_queue, p_other); }
    void printTanhLayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const { printLayer(p_queue, p_batchSize); }
};