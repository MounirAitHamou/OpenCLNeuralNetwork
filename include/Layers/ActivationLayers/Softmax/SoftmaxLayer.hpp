#pragma once

#include "Layers/ActivationLayers/ActivationLayer.hpp"

class SoftmaxLayer : public ActivationLayer {
public:

    SoftmaxLayer(const size_t p_layerId, 
                 std::shared_ptr<Utils::SharedResources> p_sharedResources,
                 const Utils::Dimensions& p_outputDimensions, 
                 const size_t p_batchSize)
        : ActivationLayer(p_layerId, p_sharedResources, p_outputDimensions, p_batchSize) 
    {
        setupKernels();
    }

    SoftmaxLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                 const H5::Group& p_layerGroup,
                 const size_t p_batchSize)
        : ActivationLayer(p_sharedResources, p_layerGroup, p_batchSize)
    {
        setupKernels();
    }

    ~SoftmaxLayer() = default;

    Utils::LayerType getType() const final override { return Utils::LayerType::Softmax; }

private:
    void setupKernels() final override;

    cl::NDRange getForwardWorkSize(const size_t p_batchSize) const final override { return cl::NDRange(p_batchSize); }

    void saveSoftmaxLayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const { saveLayer(p_layerGroup); }
    bool softmaxLayerEquals(const cl::CommandQueue& p_queue, const SoftmaxLayer& p_other) const { return layerEquals(p_queue, p_other); }
    void printSoftmaxLayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const { printLayer(p_queue, p_batchSize); }
};