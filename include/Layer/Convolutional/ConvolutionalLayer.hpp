#pragma once

#include "Layer/TrainableLayer.hpp"
#include "Utils/FilterDimensions.hpp"
#include "Utils/StrideDimensions.hpp"
#include "Utils/PaddingValues.hpp"
#include "Utils/PaddingType.hpp"

class ConvolutionalLayer : public TrainableLayer {
public:

    ConvolutionalLayer(const size_t p_layerId, 
                       std::shared_ptr<Utils::SharedResources> p_sharedResources,
                       const Utils::Dimensions& p_inputDimensions,
                       const Utils::FilterDimensions& p_filterDimensions,
                       const Utils::StrideDimensions& p_strideDimensions,
                       const Utils::PaddingType p_paddingType,
                       const Utils::ActivationType p_activationType,
                       const size_t p_batchSize,
                       std::mt19937& p_rng);
                       
    ConvolutionalLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                       const H5::Group& p_layerGroup, 
                       const size_t p_batchSize = 1);

    ~ConvolutionalLayer() = default;

    cl::Event runForward(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_inputs) override;
    cl::Event computeDeltas(const cl::CommandQueue& p_forwardBackpropQueue) override;
    void backpropDeltas(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_previousLayerDeltas, const Utils::Dimensions p_previousLayerOutputDimensions) override;
    std::pair<cl::Event, cl::Event> computeGradients(const cl::CommandQueue& p_deltaToGradientQueue, const cl::CommandQueue& p_concurrentQueue, cl::Event& p_deltaEvent, const cl::Buffer& p_inputs) override;

    Utils::LayerType getType() const override { return Utils::LayerType::Convolutional; }

    size_t getWeightsSize() const { return m_filterDimensions.getTotalElements(); }
    size_t getBiasesSize() const { return m_filterDimensions.getOutputChannels(); }

    void saveLayer(const cl::CommandQueue& p_forwardBackpropQueue, H5::Group& p_layerGroup) const override;
    bool equals(const cl::CommandQueue& p_queue, const Layer& p_other) const override;
    void print(const cl::CommandQueue& p_forwardBackpropQueue) const override;

private:
    Utils::FilterDimensions m_filterDimensions;
    Utils::StrideDimensions m_strideDimensions;
    Utils::PaddingValues m_paddingValues;
    
    size_t getInputChannels() const { return m_inputDimensions.getDimensions()[0];}
    size_t getInputHeight() const { return m_inputDimensions.getDimensions()[1];}
    size_t getInputWidth() const { return m_inputDimensions.getDimensions()[2];}

    size_t getOutputChannels() const { return m_outputDimensions.getDimensions()[0];}
    size_t getOutputHeight() const { return m_outputDimensions.getDimensions()[1];}
    size_t getOutputWidth() const { return m_outputDimensions.getDimensions()[2];}

    cl::Kernel m_im2colKernel;
    cl::Kernel m_convBiasActivationKernel;
    cl::Kernel m_convComputeOutputDeltasKernel;
    cl::Kernel m_col2imKernel;
    cl::Kernel m_convBackpropDeltasNoWeightsKernel;
    cl::Kernel m_convBackpropActivationKernel;
    cl::Kernel m_convAverageWeightsGradientsKernel;
    cl::Kernel m_convAverageBiasesGradientsKernel;
    
    cl::Buffer m_im2colBuffer;
    cl::Buffer m_col2imBuffer;

    void allocateConvolutionalLayerBuffers();
    Utils::Dimensions calculateOutputDimensions(const Utils::Dimensions& p_inputDimensions, const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, Utils::PaddingType p_paddingType) const;
    Utils::Dimensions calculateOutputDimensions() const;
    Utils::PaddingValues calculatePaddingValues(const Utils::Dimensions& p_inputDimensions, const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, const Utils::PaddingType p_paddingType) const;
    void initializeWeightsAndBiases(std::mt19937& p_rng) override;
    void setupKernels() override;
    Utils::Dimensions validateInputDimensions(const Utils::Dimensions& p_inputDimensions, const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions) const;
};