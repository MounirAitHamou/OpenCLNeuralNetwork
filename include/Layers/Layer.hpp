#pragma once

#include "Utils/Dimensions.hpp"
#include "Utils/OpenCLResources.hpp"
#include "Utils/LayerType.hpp"
#include "Visualization/LayerVisualization.hpp"
#include "Visualization/VisualizationOptions.hpp"

#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <typeinfo>
#include <stdexcept>

namespace Layers
{
    class Layer
    {
    public:
        Layer(const size_t p_layerId,
              std::shared_ptr<Utils::SharedResources> p_sharedResources,
              const Utils::Dimensions &p_outputDimensions,
              const size_t p_batchSize)
            : m_layerId(p_layerId),
              m_sharedResources(p_sharedResources),
              m_outputDimensions(p_outputDimensions)
        {
            allocateLayerBuffers(p_batchSize);
        }

        Layer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
              const H5::Group &p_layerGroup,
              const size_t p_batchSize)
            : m_sharedResources(p_sharedResources)
        {
            p_layerGroup.openAttribute("layerId").read(H5::PredType::NATIVE_HSIZE, &m_layerId);
            m_outputDimensions = Utils::Dimensions(Utils::readVectorFromHDF5<size_t>(p_layerGroup, "outputDimensions"));
            allocateLayerBuffers(p_batchSize);
        }

        virtual ~Layer() = default;

        virtual cl::Event runForward(const cl::CommandQueue &p_forwardBackpropQueue, const cl::Buffer &p_inputs, const size_t p_batchSize) = 0;
        virtual cl::Event backpropDeltas(const cl::CommandQueue &p_forwardBackpropQueue, const cl::Buffer &p_previousLayerDeltas, const size_t p_batchSize) = 0;

        virtual bool isTrainable() const { return false; }

        size_t getLayerId() const { return m_layerId; }

        cl::Buffer &getOutputs() { return m_outputs; }

        cl::Buffer &getDeltas() { return m_deltas; }

        const Utils::Dimensions &getOutputDimensions() const { return m_outputDimensions; }

        size_t getTotalOutputElements() const { return m_outputDimensions.getTotalElements(); }

        float getRandomValue(float p_min, float p_max, std::mt19937 &p_rng) const
        {
            std::uniform_real_distribution<float> distribution(p_min, p_max);
            return distribution(p_rng);
        }

        virtual Utils::LayerType getType() const = 0;
        virtual const std::vector<float> getSerializedArgs() const { return getLayerSerializedArgs(); }

        virtual void save(const cl::CommandQueue &, H5::Group &p_layerGroup) const { saveLayer(p_layerGroup); }
        virtual bool equals(const cl::CommandQueue &, const Layer &p_other) const { return layerEquals(p_other); }
        virtual void print(const cl::CommandQueue &p_queue, const size_t p_batchSize) const { printLayer(p_queue, p_batchSize); }

        size_t getBatchSize() const { return m_batchSize; }

        virtual void setBatchSize(const size_t p_batchSize)
        {
            allocateLayerBuffers(p_batchSize);
        }

        virtual Visualization::LayerVisualization buildVisualization(
            const cl::CommandQueue &p_queue,
            const Visualization::VisualizationOptions &p_options,
            size_t p_batchSize) const { return buildLayerVisualization(p_queue, p_options, p_batchSize); }

        void enqueueWriteOutputsBuffer(const cl::CommandQueue &p_queue, const std::vector<float> &p_data, const size_t p_batchSize)
        {
            if (p_data.size() != p_batchSize * getTotalOutputElements())
                throw std::invalid_argument("Data size does not match expected output buffer size");

            if (p_batchSize > getBatchSize())
                setBatchSize(p_batchSize);

            p_queue.enqueueWriteBuffer(m_outputs, BLOCKING, NO_OFFSET, p_batchSize * getTotalOutputElements() * sizeof(float), p_data.data());
        }

        void enqueueWriteDeltasBuffer(const cl::CommandQueue &p_queue, const std::vector<float> &p_data, const size_t p_batchSize)
        {
            if (p_data.size() != p_batchSize * getTotalOutputElements())
                throw std::invalid_argument("Data size does not match expected deltas buffer size");

            if (p_batchSize > getBatchSize())
                setBatchSize(p_batchSize);

            p_queue.enqueueWriteBuffer(m_deltas, BLOCKING, NO_OFFSET, p_batchSize * getTotalOutputElements() * sizeof(float), p_data.data());
        }

    protected:
        size_t m_layerId;
        std::shared_ptr<Utils::SharedResources> m_sharedResources;
        size_t m_batchSize;
        Utils::Dimensions m_outputDimensions;
        cl::Buffer m_outputs;
        cl::Buffer m_deltas;

        virtual void setupKernels() = 0;

        void allocateLayerBuffers(const size_t p_batchSize)
        {
            m_batchSize = p_batchSize;
            m_outputs = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, m_batchSize * getTotalOutputElements() * sizeof(float));
            m_deltas = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, m_batchSize * getTotalOutputElements() * sizeof(float));
        }

        std::vector<float> getLayerSerializedArgs() const
        {
            return {static_cast<float>(getType())};
        }

        void saveLayer(H5::Group &p_layerGroup) const
        {
            Utils::writeValueToHDF5<uint64_t>(p_layerGroup, "layerId", static_cast<uint64_t>(m_layerId));
            Utils::writeValueToHDF5<unsigned int>(p_layerGroup, "layerType", static_cast<unsigned int>(getType()));
            Utils::writeVectorToHDF5<size_t>(p_layerGroup, "outputDimensions", m_outputDimensions.getDimensions());
        }

        bool layerEquals(const Layer &p_other) const
        {
            return getType() == p_other.getType() &&
                   m_layerId == p_other.getLayerId() &&
                   m_outputDimensions == p_other.m_outputDimensions;
        }

        void printLayer(const cl::CommandQueue &p_queue, const size_t p_batchSize) const
        {
            std::cout << "Layer ID: " << m_layerId << "\n";
            std::cout << "Layer Type: " << Utils::layerTypeToString(getType()) << "\n";
            std::cout << "Output Dimensions: " << m_outputDimensions.toString() << "\n";
            Utils::printCLBuffer(p_queue, m_outputs, p_batchSize * getTotalOutputElements(), "Outputs");
            Utils::printCLBuffer(p_queue, m_deltas, p_batchSize * getTotalOutputElements(), "Deltas");
        }

        virtual const cl::Buffer *getBuffer(Visualization::BufferType p_type) const
        {
            switch (p_type)
            {
            case Visualization::BufferType::Outputs:
                return &m_outputs;
            case Visualization::BufferType::Deltas:
                return &m_deltas;
            default:
                return nullptr;
            }
        }

        virtual Utils::Dimensions getBufferDimensions(Visualization::BufferType p_type, size_t p_sampleCount) const
        {
            std::vector<size_t> dimsVec = {p_sampleCount};
            auto outputDimsVec = m_outputDimensions.getDimensions();
            dimsVec.insert(dimsVec.end(), outputDimsVec.begin(), outputDimsVec.end());
            switch (p_type)
            {
            case Visualization::BufferType::Outputs:
            case Visualization::BufferType::Deltas:
                return Utils::Dimensions(dimsVec);
            default:
                return {};
            }
        }

        Visualization::LayerVisualization buildLayerVisualization(
            const cl::CommandQueue &p_queue,
            const Visualization::VisualizationOptions &p_options,
            size_t p_batchSize) const
        {
            Visualization::LayerVisualization visualization;
            visualization.m_layerId = m_layerId;
            visualization.m_layerType = getType();
            visualization.m_metadata.push_back("Output Dimensions: " + m_outputDimensions.toString());
            visualization.m_metadata.push_back("Batch Size: " + std::to_string(p_batchSize));

            if (p_options.sampleIndex >= 0 &&
                static_cast<size_t>(p_options.sampleIndex) >= p_batchSize)
            {
                throw std::out_of_range("Sample index out of range for visualization");
            }

            size_t sampleCount = (p_options.sampleIndex < 0) ? p_batchSize : 1;
            size_t readOffset = (p_options.sampleIndex < 0)
                                    ? 0
                                    : static_cast<size_t>(p_options.sampleIndex) * getTotalOutputElements();

            for (auto type : p_options.m_buffersToRead)
            {
                if (auto buffer = getBuffer(type))
                {
                    auto dims = getBufferDimensions(type, sampleCount);
                    visualization.m_visualizationTensors.push_back(
                        Visualization::VisualizationTensor::createVisualizationTensor(
                            p_queue,
                            *buffer,
                            readOffset,
                            dims,
                            Visualization::bufferTypeToString(type)));
                }
            }

            return visualization;
        }
    };
}