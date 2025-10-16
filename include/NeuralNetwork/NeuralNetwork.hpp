#pragma once

#include "Utils/NetworkArgs.hpp"
#include "DataLoader/AllDataLoaders.hpp"
#include "Utils/PolicyBatch.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <future>

class NeuralNetwork {
public:
    NeuralNetwork() = default;

    NeuralNetwork(Utils::OpenCLResources&& p_oclResources, 
                  Utils::NetworkArgs p_networkArgs,
                  size_t p_seed);
    
    std::vector<float> predict(const cl::Buffer& p_inputBatch, size_t p_batchSize);
    double trainStepLoss(const Batch& p_batch, bool p_lossReporting=false);
    void trainStepPolicy(const PolicyBatch& p_batch);
    void train(DataLoader& p_dataLoader, int p_epochs, bool p_lossReporting=false);

    NeuralNetwork& addDense(const size_t p_numOutputNeurons, const Utils::ActivationType p_activationType = Utils::ActivationType::Linear);
    NeuralNetwork& addConvolutional(const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, const Utils::PaddingType p_paddingType, const Utils::ActivationType p_activationType);

    void print() const {
        std::cout << "Neural Network Details:\n";
        std::cout << "Input Dimensions: " << m_inputDimensions.toString() << "\n";
        std::cout << "Loss Function: ";
        switch (m_lossFunctionType) {
            case Utils::LossFunctionType::MeanSquaredError:
                std::cout << "Mean Squared Error\n";
                break;
            case Utils::LossFunctionType::BinaryCrossEntropy:
                std::cout << "Binary Cross Entropy\n";
                break;
            default:
                std::cout << "Unknown\n";
                break;
        }
        std::cout << "Batch Size: " << m_batchSize << "\n";
        std::cout << "Layers: \n\n";
        for(const auto& layer : m_layers){
            layer->print(m_oclResources->getForwardBackpropQueue());
        }
        
        std::cout << "Optimizer: \n\n";
        m_optimizer->print();
    }

    void saveNetwork(const std::string& p_fileName) const;
    
    static NeuralNetwork loadNetwork(std::shared_ptr<Utils::SharedResources> p_sharedResources, const std::string& p_fileName);

    bool equals(const NeuralNetwork& p_other) const;

    std::shared_ptr<Utils::SharedResources> getSharedResources() const {
        return m_oclResources->getSharedResources();
    }
private:
    std::vector<std::unique_ptr<Layer>> m_layers;

    size_t m_batchSize;
    Utils::Dimensions m_inputDimensions;

    std::unique_ptr<Utils::OpenCLResources> m_oclResources;

    std::unique_ptr<Optimizer> m_optimizer;

    Utils::LossFunctionType m_lossFunctionType;

    cl::Kernel m_lossGradientKernel;
    cl::Kernel m_policyGradientKernel;

    std::mt19937 m_rng;

    NeuralNetwork(Utils::OpenCLResources&& p_oclResources, const H5::H5File& p_file);

    double computeLossAsync(cl::Event& p_forwardEvent, const std::vector<float>& p_batchTargets);

    cl::Event forward(const cl::Buffer& p_batchInputs, size_t p_batchSize);
    void computeLossGradients(const cl::Buffer& p_batchTargets);
    void computePolicyGradients(const cl::Buffer& p_batchActions, const cl::Buffer& p_batchRewards);
    void backward(const cl::Buffer& p_batchInputs);
    void setupKernels();
};