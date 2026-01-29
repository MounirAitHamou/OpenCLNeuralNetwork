#pragma once

#include "NeuralNetworks/NeuralNetwork.hpp"

namespace NeuralNetworks::Local {
    class LocalNeuralNetwork: public NeuralNetwork {
    public:
        LocalNeuralNetwork() = default;
        
        LocalNeuralNetwork(Utils::OpenCLResources&& p_oclResources, 
                            const Utils::NetworkArgs& p_networkArgs,
                            const size_t p_seed,
                            const size_t p_batchSize);
        
        std::vector<float> predict(const cl::Buffer& p_inputBatch, size_t p_batchSize);
        double trainStep(const Utils::Batch& p_batch, bool p_lossReporting=false);
        void train(DataLoaders::DataLoader& p_dataLoader, int p_epochs, bool p_lossReporting=false);
        cl::Event forward(const cl::Buffer& p_batchInputs, size_t p_batchSize);
        double computeLossAsync(cl::Event& p_forwardEvent, const std::vector<float>& p_batchTargets, const size_t p_batchSize);
        cl::Event computeLossGradients(const cl::Buffer& p_batchTargets, const size_t p_batchSize);
        void uploadOutputDeltas(const std::vector<float>& p_hostGradients);
        void copyOutputDeltasFromBuffer(const cl::Buffer& p_deviceGradients, const size_t p_batchSize);
        void backward(cl::Event& p_deltaEvent, const cl::Buffer& p_batchInputs, const size_t p_batchSize);
        
        LocalNeuralNetwork& addDense(const size_t p_numOutputNeurons);
        LocalNeuralNetwork& addConvolutional(const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, const Utils::PaddingType p_paddingType);
        LocalNeuralNetwork& addLeakyReLU(float p_alpha);
        LocalNeuralNetwork& addReLU();
        LocalNeuralNetwork& addSigmoid();
        LocalNeuralNetwork& addTanh();
        LocalNeuralNetwork& addSoftmax();
        
        void save(const std::string& p_fileName) const;
        static LocalNeuralNetwork load(std::shared_ptr<Utils::SharedResources> p_sharedResources, const std::string& p_fileName, const size_t p_batchSize);
        bool equals(const LocalNeuralNetwork& p_other) const;
        void print() const;
        
        std::shared_ptr<Utils::SharedResources> getSharedResources() const {
            return m_oclResources->getSharedResources();
        }

        void setBatchSize(const size_t p_batchSize) override {
            m_batchSize = p_batchSize;
            for (auto& layer : m_layers) {
                layer->setBatchSize(p_batchSize);
            }
        }

        Utils::NetworkType getType() const final override { return Utils::NetworkType::Local; }


    private:
        std::unique_ptr<Optimizers::Optimizer> m_optimizer;
        std::mt19937 m_rng;
        
        LocalNeuralNetwork(Utils::OpenCLResources&& p_oclResources, const H5::H5File& p_file, const size_t p_batchSize);
    };
}