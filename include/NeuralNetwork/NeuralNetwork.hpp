#pragma once

#include "Utils/NetworkArgs.hpp"
#include "DataLoaders/AllDataLoaders.hpp"
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
    void train(DataLoader& p_dataLoader, int p_epochs, bool p_lossReporting=false);
    cl::Event forward(const cl::Buffer& p_batchInputs, size_t p_batchSize);
    double computeLossAsync(cl::Event& p_forwardEvent, const std::vector<float>& p_batchTargets, const size_t p_batchSize);
    void computeLossGradients(const cl::Buffer& p_batchTargets, const size_t p_batchSize);
    void uploadOutputDeltas(const std::vector<float>& p_hostGradients);
    void copyOutputDeltasFromBuffer(const cl::Buffer& p_deviceGradients, const size_t p_batchSize);
    void backward(const cl::Buffer& p_batchInputs, const size_t p_batchSize);

    NeuralNetwork& addDense(const size_t p_numOutputNeurons);
    NeuralNetwork& addConvolutional(const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, const Utils::PaddingType p_paddingType);
    NeuralNetwork& addLeakyReLU(float p_alpha);
    NeuralNetwork& addReLU();
    NeuralNetwork& addSigmoid();
    NeuralNetwork& addSoftmax();
    NeuralNetwork& addTanh();

    void save(const std::string& p_fileName) const;
    static NeuralNetwork load(std::shared_ptr<Utils::SharedResources> p_sharedResources, const std::string& p_fileName);
    bool equals(const NeuralNetwork& p_other) const;
    void print() const;

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

    
    void setupKernels();
};