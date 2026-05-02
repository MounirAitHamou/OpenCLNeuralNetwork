#pragma once

#include "Layers/Layer.hpp"
namespace Layers::Activation
{
    class ActivationLayer : public Layer
    {
    public:
        ActivationLayer(const size_t p_layerId,
                        std::shared_ptr<Utils::SharedResources> p_sharedResources,
                        const Utils::Dimensions &p_outputDimensions,
                        const size_t p_batchSize)
            : Layer(p_layerId, p_sharedResources, p_outputDimensions, p_batchSize)
        {
        }

        ActivationLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                        const H5::Group &p_layerGroup,
                        const size_t p_batchSize)
            : Layer(p_sharedResources, p_layerGroup, p_batchSize)
        {
        }

        ~ActivationLayer() = default;

        cl::Event runForward(const cl::CommandQueue &p_forwardBackpropQueue, const cl::Buffer &p_inputs, const size_t p_batchSize) override
        {
            if (m_batchSize < p_batchSize)
                setBatchSize(p_batchSize);
            cl::Event forwardEvent;

            Utils::setKernelArgs(m_forwardKernel, p_inputs);

            cl_int err = p_forwardBackpropQueue.enqueueNDRangeKernel(
                m_forwardKernel,
                cl::NullRange,
                getForwardWorkSize(p_batchSize),
                cl::NullRange,
                nullptr,
                &forwardEvent);

            if (err != CL_SUCCESS)
            {
                CLNN_FATAL("Failed to enqueue forward kernel");
            }

            return forwardEvent;
        }

        cl::Event backpropDeltas(const cl::CommandQueue &p_forwardBackpropQueue, const cl::Buffer &p_previousLayerDeltas, const size_t p_batchSize) override
        {
            if (m_batchSize < p_batchSize)
                setBatchSize(p_batchSize);
            cl::Event backpropEvent;
            Utils::setKernelArgs(m_backwardKernel, p_previousLayerDeltas);

            cl_int err = p_forwardBackpropQueue.enqueueNDRangeKernel(
                m_backwardKernel,
                cl::NullRange,
                getForwardWorkSize(p_batchSize),
                cl::NullRange,
                nullptr,
                &backpropEvent);

            if (err != CL_SUCCESS)
            {
                CLNN_FATAL("Failed to enqueue backward kernel");
            }

            return backpropEvent;
        }

        virtual void setBatchSize(const size_t p_batchSize) override
        {
            allocateLayerBuffers(p_batchSize);
            Utils::setKernelArgs(1, m_forwardKernel, getOutputs());
            Utils::setKernelArgs(1, m_backwardKernel, getDeltas(), getOutputs());
        }

        Visualization::LayerVisualization buildVisualization(const cl::CommandQueue &p_queue, const Visualization::VisualizationOptions &p_options, size_t p_batchSize) const override { return buildActivationLayerVisualization(p_queue, p_options, p_batchSize); }

    protected:
        cl::Kernel m_forwardKernel;
        cl::Kernel m_backwardKernel;

        virtual cl::NDRange getForwardWorkSize(const size_t p_batchSize) const { return cl::NDRange(p_batchSize * getTotalOutputElements()); }

        void saveActivationLayer(H5::Group &p_layerGroup) const { saveLayer(p_layerGroup); }

        bool activationLayerEquals(const ActivationLayer &p_other) const { return layerEquals(p_other); }

        void printActivationLayer(const cl::CommandQueue &p_queue, const size_t p_batchSize) const { printLayer(p_queue, p_batchSize); }

        Visualization::LayerVisualization buildActivationLayerVisualization(const cl::CommandQueue &p_queue, const Visualization::VisualizationOptions &p_options, size_t p_batchSize) const
        {
            Visualization::LayerVisualization visualization = buildLayerVisualization(p_queue, p_options, p_batchSize);
            return visualization;
        }
    };
}