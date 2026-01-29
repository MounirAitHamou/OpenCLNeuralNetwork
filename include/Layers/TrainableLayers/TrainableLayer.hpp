#pragma once

#include "Layers/Layer.hpp"
#include <clblast.h>
#include <utility>

namespace Layers::Trainable {
    class TrainableLayer : public Layer {
    public:
        
        TrainableLayer(const size_t p_layerId,
                    std::shared_ptr<Utils::SharedResources> p_sharedResources,
                    const Utils::Dimensions& p_inputDimensions,
                    const Utils::Dimensions& p_outputDimensions,
                    const size_t p_batchSize)
            : Layer(p_layerId, p_sharedResources, p_outputDimensions, p_batchSize),
            m_inputDimensions(p_inputDimensions)
        {}

        TrainableLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                    const H5::Group& p_layerGroup,
                    const size_t p_batchSize)
            : Layer(p_sharedResources, p_layerGroup, p_batchSize)
            { m_inputDimensions = Utils::Dimensions(Utils::readVectorFromHDF5<size_t>(p_layerGroup, "inputDimensions")); }

        virtual ~TrainableLayer() = default;
        
        virtual std::pair<cl::Event, cl::Event> computeGradients(const cl::CommandQueue& p_deltaToGradientQueue, cl::Event& p_backpropEvent, const cl::Buffer& p_inputs, const size_t p_batchSize) = 0;
        
        
        bool isTrainable() const override { return true; }
        
        size_t getTotalInputElements() const { return m_inputDimensions.getTotalElements(); }
        
        const Utils::Dimensions& getInputDimensions() const { return m_inputDimensions; }
        
        cl::Buffer& getWeights() { return m_weights; }
        cl::Buffer& getBiases() { return m_biases; }

        cl::Event& setWeights(const cl::CommandQueue& p_queue, const std::vector<cl::Event>& p_waitList, const std::vector<float>& p_weightsVec) {
            if (p_weightsVec.size() != getWeightsSize()) {
                throw std::invalid_argument("Input weights vector size does not match layer weights size.");
            }

            cl_int err;
            cl::Event writeEvent;
            err = p_queue.enqueueWriteBuffer(
                m_weights,
                NON_BLOCKING_READ,
                NO_OFFSET,
                p_weightsVec.size() * sizeof(float),
                p_weightsVec.data(),
                &p_waitList,
                &writeEvent
            );
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to enqueue write buffer for weights. Error code: " + std::to_string(err));
            }
            return writeEvent;
        }
        cl::Event& setBiases(const cl::CommandQueue& p_queue, const std::vector<cl::Event>& p_waitList, const std::vector<float>& p_biasesVec) {
            if (p_biasesVec.size() != getBiasesSize()) {
                throw std::invalid_argument("Input biases vector size does not match layer biases size.");
            }

            cl_int err;
            cl::Event writeEvent;
            err = p_queue.enqueueWriteBuffer(
                m_biases,
                NON_BLOCKING_READ,
                NO_OFFSET,
                p_biasesVec.size() * sizeof(float),
                p_biasesVec.data(),
                &p_waitList,
                &writeEvent
            );
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to enqueue write buffer for biases. Error code: " + std::to_string(err));
            }
            return writeEvent;
        }

        std::vector<float> getWeightsCPU(const cl::CommandQueue& p_queue) const {
            return Utils::readCLBuffer(p_queue, m_weights, getWeightsSize());
        }

        std::vector<float> getBiasesCPU(const cl::CommandQueue& p_queue) const {
            return Utils::readCLBuffer(p_queue, m_biases, getBiasesSize());
        }
        
        cl::Buffer& getWeightsGradients() { return m_weightsGradients; }
        cl::Buffer& getBiasesGradients() { return m_biasesGradients; }
        
        virtual size_t getWeightsSize() const = 0;
        virtual size_t getBiasesSize() const = 0;
        
        
        virtual void save(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const override { saveTrainableLayer(p_queue, p_layerGroup); }
        virtual bool equals(const cl::CommandQueue& p_queue, const Layer& p_other) const override { return trainableLayerEquals(p_queue, p_other); }
        virtual void print(const cl::CommandQueue& p_queue, const size_t p_batchSize) const override { printTrainableLayer(p_queue, p_batchSize); }
    protected:
        Utils::Dimensions m_inputDimensions;
        
        cl::Buffer m_weights;
        cl::Buffer m_biases;
        cl::Buffer m_weightsGradients;
        cl::Buffer m_biasesGradients;
        
        cl::Kernel m_biasKernel;
        
        virtual void initializeWeightsAndBiases(std::mt19937& p_rng) = 0;
        
        void setupTrainableKernels() {}
        
        void saveTrainableLayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const {
            saveLayer(p_layerGroup);
            Utils::writeVectorToHDF5<size_t>(p_layerGroup, "inputDimensions", m_inputDimensions.getDimensions());
            Utils::saveBuffer(p_queue, m_weights, p_layerGroup, "weights", getWeightsSize());
            Utils::saveBuffer(p_queue, m_biases, p_layerGroup, "biases", getBiasesSize());
        }
        
        bool trainableLayerEquals(const cl::CommandQueue& p_queue, const Layer& p_other) const {
            if (!layerEquals(p_queue, p_other)) return false;
            const TrainableLayer& otherTrainable = static_cast<const TrainableLayer&>(p_other);
            return m_inputDimensions == otherTrainable.m_inputDimensions && 
                Utils::compareCLBuffers(p_queue, m_weights, otherTrainable.m_weights, getWeightsSize()) &&
                Utils::compareCLBuffers(p_queue, m_biases, otherTrainable.m_biases, getBiasesSize());
        }
        
        void printTrainableLayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const {
            printLayer(p_queue, p_batchSize);
            std::cout << "Input Dimensions: " << m_inputDimensions.toString() << "\n";
            std::cout << "Weights Size: " << getWeightsSize() << "\n";
            std::cout << "Biases Size: " << getBiasesSize() << "\n";
            Utils::printCLBuffer(p_queue, m_weights, getWeightsSize(), "Weights");
            Utils::printCLBuffer(p_queue, m_biases, getBiasesSize(), "Biases");
            Utils::printCLBuffer(p_queue, m_weightsGradients, getWeightsSize(), "Weight Gradients");
            Utils::printCLBuffer(p_queue, m_biasesGradients, getBiasesSize(), "Bias Gradients");
        }
    };
}