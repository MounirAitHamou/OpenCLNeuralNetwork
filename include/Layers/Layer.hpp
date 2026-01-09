#pragma once

#include "Utils/Dimensions.hpp"
#include "Utils/OpenCLResources.hpp"
#include "Utils/LayerType.hpp"

#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <typeinfo>


namespace Layers {
    class Layer {
    public:

        Layer(const size_t p_layerId, 
            std::shared_ptr<Utils::SharedResources> p_sharedResources,
            const Utils::Dimensions& p_outputDimensions, 
            const size_t p_batchSize)
            : m_sharedResources(p_sharedResources),
            m_layerId(p_layerId), 
            m_outputDimensions(p_outputDimensions) {
            allocateLayerBuffers(p_batchSize);
            }

        Layer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
            const H5::Group& p_layerGroup,
            const size_t p_batchSize)
            : m_sharedResources(p_sharedResources) {
            p_layerGroup.openAttribute("layerId").read(H5::PredType::NATIVE_HSIZE, &m_layerId);
            m_outputDimensions = Utils::Dimensions(Utils::readVectorFromHDF5<size_t>(p_layerGroup, "outputDimensions"));
            allocateLayerBuffers(p_batchSize);
            }

        virtual ~Layer() = default;

        virtual cl::Event runForward(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_inputs, const size_t p_batchSize) = 0;
        virtual cl::Event backpropDeltas(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_previousLayerDeltas, const size_t p_batchSize) = 0;

        virtual bool isTrainable() const { return false; }

        size_t getLayerId() const { return m_layerId; }

        cl::Buffer& getOutputs() { return m_outputs; }

        cl::Buffer& getDeltas() { return m_deltas; }

        const Utils::Dimensions& getOutputDimensions() const { return m_outputDimensions; }

        size_t getTotalOutputElements() const { return m_outputDimensions.getTotalElements(); }

        float getRandomValue(float p_min, float p_max, std::mt19937& p_rng) const {
            std::uniform_real_distribution<float> distribution(p_min, p_max);
            return distribution(p_rng);
        }

        virtual Utils::LayerType getType() const = 0;
        virtual const std::vector<float> getSerializedArgs() const { return getLayerSerializedArgs(); }

        virtual void save(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const { saveLayer(p_layerGroup); }
        virtual bool equals(const cl::CommandQueue& p_queue, const Layer& p_other) const { return layerEquals(p_queue, p_other); }
        virtual void print(const cl::CommandQueue& p_queue, const size_t p_batchSize) const { printLayer(p_queue, p_batchSize); }

        virtual void setBatchSize(const size_t p_batchSize) {
            allocateLayerBuffers(p_batchSize);
        }
    protected:

        size_t m_layerId;
        size_t m_batchSize;

        Utils::Dimensions m_outputDimensions;

        cl::Buffer m_outputs;
        cl::Buffer m_deltas;

        std::shared_ptr<Utils::SharedResources> m_sharedResources;

        virtual void setupKernels() = 0;

        void allocateLayerBuffers(const size_t p_batchSize) {
            m_outputs    = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, p_batchSize * getTotalOutputElements() * sizeof(float));
            m_deltas     = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, p_batchSize * getTotalOutputElements() * sizeof(float));
            m_batchSize = p_batchSize;
        }

        std::vector<float> getLayerSerializedArgs() const { return {static_cast<float>(getType())}; }

        void saveLayer(H5::Group& p_layerGroup) const {
            Utils::writeValueToHDF5<size_t>(p_layerGroup, "layerId", m_layerId);
            Utils::writeValueToHDF5<unsigned int>(p_layerGroup, "layerType", static_cast<unsigned int>(getType()));
            Utils::writeVectorToHDF5<size_t>(p_layerGroup, "outputDimensions", m_outputDimensions.getDimensions());
        }

        bool layerEquals(const cl::CommandQueue& p_queue, const Layer& p_other) const {
            return getType() == p_other.getType() && 
                m_layerId == p_other.getLayerId() && 
                m_outputDimensions == p_other.m_outputDimensions;
        }

        void printLayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const {
            std::cout << "Layer ID: " << m_layerId << "\n";
            std::cout << "Layer Type: " << Utils::layerTypeToString(getType()) << "\n";
            std::cout << "Output Dimensions: " << m_outputDimensions.toString() << "\n";
            Utils::printCLBuffer(p_queue, m_outputs, p_batchSize * getTotalOutputElements(), "Outputs");
            Utils::printCLBuffer(p_queue, m_deltas, p_batchSize * getTotalOutputElements(), "Deltas");
        }

        
    };
}
