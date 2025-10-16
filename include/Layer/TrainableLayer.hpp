#pragma once

#include "Layer/Layer.hpp"
#include "Utils/ActivationType.hpp"
#include <clblast.h>
#include <utility>

class TrainableLayer : public Layer {
public:
    
    TrainableLayer(const size_t p_layerId,
                   std::shared_ptr<Utils::SharedResources> p_sharedResources,
                   const Utils::Dimensions& p_inputDimensions,
                   const Utils::Dimensions& p_outputDimensions,
                   const Utils::ActivationType p_activationType = Utils::ActivationType::Linear,
                   const size_t p_batchSize = 1)
        : Layer(p_layerId, p_sharedResources, p_inputDimensions, p_outputDimensions, p_batchSize), m_activationType(p_activationType) 
    {}

    TrainableLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                   const H5::Group& p_layerGroup,
                   const size_t p_batchSize)
        : Layer(p_sharedResources, p_layerGroup, p_batchSize)
    {
        unsigned int activationTypeUInt;
        p_layerGroup.openAttribute("activationType").read(H5::PredType::NATIVE_UINT, &activationTypeUInt);
        m_activationType = Utils::activationTypeFromUint(activationTypeUInt);
    }

    
    virtual std::pair<cl::Event, cl::Event> computeGradients(const cl::CommandQueue& p_deltaToGradientQueue, const cl::CommandQueue& p_concurrentQueue, cl::Event& p_deltaEvent, const cl::Buffer& p_inputs) = 0;

    bool isTrainable() const override { return true; }

    cl::Buffer& getWeights() { return m_weights; }
    cl::Buffer& getBiases() { return m_biases; }

    cl::Buffer& getWeightsGradients() { return m_weightsGradients; }
    cl::Buffer& getBiasesGradients() { return m_biasesGradients; }

    cl::Buffer& getPreActivations() { return m_preActivations; }

    virtual size_t getWeightsSize() const = 0;

    virtual size_t getBiasesSize() const = 0;

    virtual ~TrainableLayer() = default;

protected:
    cl::Buffer m_weights;
    cl::Buffer m_biases;
    cl::Buffer m_weightsGradients;
    cl::Buffer m_biasesGradients;
    cl::Buffer m_preActivations;
    cl::Buffer m_clblastWorkspace;
    cl::Buffer m_clblastDeltaWorkspace;
    cl::Buffer m_onesBuffer;
    
    Utils::ActivationType m_activationType;

    Utils::ActivationType getActivationType() { return m_activationType; }

    virtual void initializeWeightsAndBiases(std::mt19937& p_rng) = 0;
};