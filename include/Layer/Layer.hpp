#pragma once

#include "Utils/Dimensions.hpp"
#include "Utils/OpenCLResources.hpp"
#include "Utils/LossFunctionType.hpp"
#include "Utils/LayerType.hpp"

#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <typeinfo>

class Layer {
public:

    Layer(const size_t p_layerId, 
          std::shared_ptr<Utils::SharedResources> p_sharedResources,
          const Utils::Dimensions& p_inputDimensions, 
          const Utils::Dimensions& p_outputDimensions, 
          const size_t p_batchSize = 1)
        : m_layerId(p_layerId), m_sharedResources(p_sharedResources), m_inputDimensions(p_inputDimensions), 
          m_outputDimensions(p_outputDimensions), m_batchSize(p_batchSize), m_maxBatchSize(p_batchSize) {
            allocateLayerBuffers();
        }

    Layer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
          const H5::Group& p_layerGroup,
          const size_t p_batchSize)
        : m_sharedResources(p_sharedResources), m_batchSize(p_batchSize), m_maxBatchSize(p_batchSize) {
            p_layerGroup.openAttribute("layerId").read(H5::PredType::NATIVE_HSIZE, &m_layerId);
        }

    virtual ~Layer() = default;

    virtual cl::Event runForward(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_inputs) = 0;
    virtual cl::Event computeDeltas(const cl::CommandQueue& p_forwardBackpropQueue) = 0;
    virtual void backpropDeltas(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_previousLayerDeltas, const Utils::Dimensions p_previousLayerOutputDimensions) = 0;

    virtual bool isTrainable() const { return false; }

    size_t getLayerId() const { return m_layerId; }

    cl::Buffer& getOutputs() {
        return m_outputs;
    }

    cl::Buffer& getDeltas() {
        return m_deltas;
    }

    void setBatchSize(size_t p_batchSize) {
        m_batchSize = p_batchSize;
    }

    size_t getBatchSize(){
        return m_batchSize;
    }

    const Utils::Dimensions& getInputDimensions() const {
        return m_inputDimensions;
    }

    const Utils::Dimensions& getOutputDimensions() const {
        return m_outputDimensions;
    }

    size_t getTotalOutputElements() const {
        return m_outputDimensions.getTotalElements();
    }

    size_t getTotalInputElements() const {
        return m_inputDimensions.getTotalElements();
    }

    float getRandomValue(float p_min, float p_max, std::mt19937& p_rng) const {
        std::uniform_real_distribution<float> distribution(p_min, p_max);
        return distribution(p_rng);
    }

    virtual Utils::LayerType getType() const = 0;

    virtual void saveLayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const = 0;
    virtual bool equals(const cl::CommandQueue& p_queue, const Layer& p_other) const = 0;

    virtual void print(const cl::CommandQueue& p_queue) const = 0;


protected:

    size_t m_layerId;

    Utils::Dimensions m_inputDimensions;
    Utils::Dimensions m_outputDimensions;

    size_t m_batchSize;
    size_t m_maxBatchSize;

    cl::Buffer m_outputs;
    cl::Buffer m_deltas;

    std::shared_ptr<Utils::SharedResources> m_sharedResources;

    virtual void setupKernels() = 0;

protected:
    void allocateLayerBuffers() {
        m_outputs    = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, m_batchSize * getTotalOutputElements() * sizeof(float));
    }
};