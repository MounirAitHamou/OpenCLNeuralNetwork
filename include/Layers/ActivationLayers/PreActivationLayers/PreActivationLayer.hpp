#pragma once

#include "Layers/ActivationLayers/ActivationLayer.hpp"

class PreActivationLayer : public ActivationLayer {
public:
    PreActivationLayer(const size_t p_layerId,
                       std::shared_ptr<Utils::SharedResources> p_sharedResources,
                       const Utils::Dimensions& p_outputDimensions,
                       const size_t p_batchSize)
        : ActivationLayer(p_layerId, p_sharedResources, p_outputDimensions, p_batchSize) 
    {
        allocatePreActivationLayerBuffers(p_batchSize);
    }

    PreActivationLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                       const H5::Group& p_layerGroup,
                       const size_t p_batchSize)
        : ActivationLayer(p_sharedResources, p_layerGroup, p_batchSize)
    {
        allocatePreActivationLayerBuffers(p_batchSize);
    }

    virtual ~PreActivationLayer() = default;

    cl::Buffer& getPreActivations() { return m_preActivations; }

    virtual void print(const cl::CommandQueue& p_queue, const size_t p_batchSize) const override { printPreActivationLayer(p_queue, p_batchSize); }

protected:
    cl::Buffer m_preActivations;

    void allocatePreActivationLayerBuffers(const size_t p_batchSize) {
        m_preActivations = cl::Buffer(
            m_sharedResources->getContext(), 
            CL_MEM_READ_WRITE, 
            p_batchSize * getTotalOutputElements() * sizeof(float)
        );
    }

    void savePreActivationLayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const { saveLayer(p_layerGroup); }

    bool preActivationLayerEquals(const cl::CommandQueue& p_queue, const PreActivationLayer& p_other) const { return layerEquals(p_queue, p_other); }

    void printPreActivationLayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const {
        printLayer(p_queue, p_batchSize);
        Utils::printCLBuffer(p_queue, m_preActivations, p_batchSize * m_outputDimensions.getTotalElements(), "Pre Activations");
    }
};