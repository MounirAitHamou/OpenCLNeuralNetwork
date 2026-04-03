#pragma once

#include "Layers/ActivationLayers/ActivationLayer.hpp"
namespace Layers::Activation
{
    class PreActivationLayer : public ActivationLayer
    {
    public:
        PreActivationLayer(const size_t p_layerId,
                           std::shared_ptr<Utils::SharedResources> p_sharedResources,
                           const Utils::Dimensions &p_outputDimensions,
                           const size_t p_batchSize)
            : ActivationLayer(p_layerId, p_sharedResources, p_outputDimensions, p_batchSize)
        {
            allocatePreActivationLayerBuffers(p_batchSize);
        }

        PreActivationLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                           const H5::Group &p_layerGroup,
                           const size_t p_batchSize)
            : ActivationLayer(p_sharedResources, p_layerGroup, p_batchSize)
        {
            allocatePreActivationLayerBuffers(p_batchSize);
        }

        virtual ~PreActivationLayer() = default;

        cl::Buffer &getPreActivations() { return m_preActivations; }

        virtual void print(const cl::CommandQueue &p_queue, const size_t p_batchSize) const override { printPreActivationLayer(p_queue, p_batchSize); }

        void setBatchSize(const size_t p_batchSize) final override
        {
            allocateLayerBuffers(p_batchSize);
            allocatePreActivationLayerBuffers(p_batchSize);
            Utils::setKernelArgs(1, m_forwardKernel, getOutputs(), getPreActivations());
            Utils::setKernelArgs(1, m_backwardKernel, getDeltas(), getPreActivations());
        }

        Visualization::LayerVisualization buildVisualization(const cl::CommandQueue &p_queue, const Visualization::VisualizationOptions &p_options, size_t p_batchSize) const override { return buildPreActivationLayerVisualization(p_queue, p_options, p_batchSize); }

    protected:
        cl::Buffer m_preActivations;

        void allocatePreActivationLayerBuffers(const size_t p_batchSize)
        {
            m_preActivations = cl::Buffer(
                m_sharedResources->getContext(),
                CL_MEM_READ_WRITE,
                p_batchSize * getTotalOutputElements() * sizeof(float));
        }

        void savePreActivationLayer(H5::Group &p_layerGroup) const { saveLayer(p_layerGroup); }

        bool preActivationLayerEquals(const PreActivationLayer &p_other) const { return layerEquals(p_other); }

        void printPreActivationLayer(const cl::CommandQueue &p_queue, const size_t p_batchSize) const
        {
            printLayer(p_queue, p_batchSize);
            Utils::printCLBuffer(p_queue, m_preActivations, p_batchSize * m_outputDimensions.getTotalElements(), "Pre Activations");
        }

        const cl::Buffer *getBuffer(Visualization::BufferType p_type) const override
        {
            switch (p_type)
            {
            case Visualization::BufferType::PreActivations:
                return &m_preActivations;
            default:
                return Layer::getBuffer(p_type);
            }
        }

        Utils::Dimensions getBufferDimensions(Visualization::BufferType p_type, size_t p_sampleCount) const override
        {
            std::vector<size_t> dimsVec = {p_sampleCount};
            auto outputDimsVec = m_outputDimensions.getDimensions();
            dimsVec.insert(dimsVec.end(), outputDimsVec.begin(), outputDimsVec.end());
            switch (p_type)
            {
            case Visualization::BufferType::PreActivations:
                return Utils::Dimensions(dimsVec);
            default:
                return Layer::getBufferDimensions(p_type, p_sampleCount);
            }
        }

        Visualization::LayerVisualization buildPreActivationLayerVisualization(const cl::CommandQueue &p_queue, const Visualization::VisualizationOptions &p_options, size_t p_batchSize) const
        {
            Visualization::LayerVisualization visualization = buildActivationLayerVisualization(p_queue, p_options, p_batchSize);
            return visualization;
        }
    };
}