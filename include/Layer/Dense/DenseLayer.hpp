#pragma once

#include "Layer/TrainableLayer.hpp"

class DenseLayer : public TrainableLayer {
public:

    DenseLayer(const size_t p_layerId, 
               std::shared_ptr<Utils::SharedResources> p_sharedResources,
               const Utils::Dimensions& p_inputDimensions, 
               const Utils::Dimensions& p_outputDimensions,
               const Utils::ActivationType p_activationType,
               const size_t p_batchSize,
                std::mt19937& p_rng);
               
    DenseLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
               const H5::Group& p_layerGroup, 
               const size_t p_batchSize = 1);

    ~DenseLayer() = default;

    cl::Event runForward(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_inputs) override;
    cl::Event computeDeltas(const cl::CommandQueue& p_forwardBackpropQueue) override;
    void backpropDeltas(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_previousLayerDeltas, const Utils::Dimensions p_previousLayerOutputDimensions) override;
    std::pair<cl::Event, cl::Event> computeGradients(const cl::CommandQueue& p_deltaToGradientQueue, const cl::CommandQueue& p_concurrentQueue, cl::Event& p_deltaEvent, const cl::Buffer& p_inputs) override;

    Utils::LayerType getType() const override {
        return Utils::LayerType::Dense;
    }

    size_t getWeightsSize() const {
        return getTotalInputElements() * getTotalOutputElements();
    };

    size_t getBiasesSize() const {
        return getTotalOutputElements();
    }

    void saveLayer(const cl::CommandQueue& p_forwardBackpropQueue, H5::Group& p_layerGroup) const override;
    bool equals(const cl::CommandQueue& p_queue, const Layer& p_other) const override;

    void print(const cl::CommandQueue& p_forwardBackpropQueue) const override;

private:
    cl::Kernel m_denseBiasActivationKernel;
    cl::Kernel m_denseComputeOutputDeltasKernel;
    cl::Kernel m_denseBackpropDeltasNoWeightsKernel;
    cl::Kernel m_denseBackpropActivationKernel;
    cl::Kernel m_denseAverageWeightGradientsKernel;
    cl::Kernel m_denseAverageBiasGradientsKernel;
    
    cl::Buffer m_onesBuffer;

    void allocateDenseLayerBuffers();
    void initializeWeightsAndBiases(std::mt19937& p_rng) override;
    void setupKernels() override;
};