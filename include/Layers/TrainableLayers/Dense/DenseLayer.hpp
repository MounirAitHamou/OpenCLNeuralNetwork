#pragma once

#include "Layers/TrainableLayers/TrainableLayer.hpp"
namespace Layers::Trainable {
    class DenseLayer : public TrainableLayer {
    public:
        DenseLayer(const size_t p_layerId, 
                std::shared_ptr<Utils::SharedResources> p_sharedResources,
                const Utils::Dimensions& p_inputDimensions, 
                const Utils::Dimensions& p_outputDimensions,
                const size_t p_batchSize,
                std::mt19937& p_rng);
                
        DenseLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                const H5::Group& p_layerGroup, 
                const size_t p_batchSize);

        ~DenseLayer() = default;

        cl::Event runForward(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_inputs, const size_t p_batchSize) final override;
        cl::Event backpropDeltas(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_previousLayerDeltas, const size_t p_batchSize) final override;
        std::pair<cl::Event, cl::Event> computeGradients(const cl::CommandQueue& p_deltaToGradientQueue, cl::Event& p_backpropEvent, const cl::Buffer& p_inputs, const size_t p_batchSize) final override;

        Utils::LayerType getType() const final override { return Utils::LayerType::Dense; }

        size_t getWeightsSize() const final override { return getTotalInputElements() * getTotalOutputElements(); };
        size_t getBiasesSize() const final override { return getTotalOutputElements(); }

        cl::Buffer& getOnesBuffer() { return m_onesBuffer; }

        cl::Buffer& getclblastWorkspace() { return m_clblastWorkspace; }
        cl::Buffer& getclblastDeltaWorkspace() { return m_clblastDeltaWorkspace; }

        const std::vector<float> getSerializedArgs() const final override {
            std::vector<float> layerArgs = getLayerSerializedArgs();
            layerArgs.push_back(static_cast<float>(getTotalOutputElements()));
            return layerArgs;  
        }

        void setBatchSize(const size_t p_batchSize) final override {
            allocateLayerBuffers(p_batchSize);
            m_biasKernel.setArg(1, getOutputs());

            m_onesBuffer       = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, p_batchSize * sizeof(float), std::vector<float>(p_batchSize, 1.0f).data());
            
            size_t flatInputSize = m_inputDimensions.getTotalElements();
            size_t flatOutputSize = m_outputDimensions.getTotalElements();

            m_clblastWorkspace = cl::Buffer(
                m_sharedResources->getContext(),
                CL_MEM_READ_WRITE,
                std::max({p_batchSize * flatOutputSize, flatOutputSize * flatInputSize}) * sizeof(float)
            );

            m_clblastDeltaWorkspace = cl::Buffer(
                m_sharedResources->getContext(),
                CL_MEM_READ_WRITE,
                std::max({p_batchSize * flatInputSize, flatInputSize * flatOutputSize}) * sizeof(float)
            );
        }

    private:
        cl::Buffer m_onesBuffer;
        cl::Buffer m_clblastWorkspace;
        cl::Buffer m_clblastDeltaWorkspace;

        void allocateDenseLayerBuffers(const size_t p_batchSize);
        void initializeWeightsAndBiases(std::mt19937& p_rng) final override;
        void setupKernels() final override;
        
        void saveDenseLayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const { saveTrainableLayer(p_queue, p_layerGroup); }
        bool denseLayerEquals(const cl::CommandQueue& p_queue, const Layer& p_other) const { return trainableLayerEquals(p_queue, p_other); }
        void printDenseLayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const { printTrainableLayer(p_queue, p_batchSize); }
    };
}