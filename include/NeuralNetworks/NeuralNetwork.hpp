#pragma once

#include "DataLoaders/AllDataLoaders.hpp"
#include "Utils/NetworkArgs.hpp"
#include "Utils/NetworkType.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <future>

class NeuralNetwork {
public:
    NeuralNetwork() = default;

    NeuralNetwork(Utils::OpenCLResources&& p_oclResources, const Utils::Dimensions& p_inputDimensions, const size_t p_batchSize) 
    : m_batchSize(p_batchSize),
      m_inputDimensions(p_inputDimensions)
      {
        m_oclResources = std::make_unique<Utils::OpenCLResources>(std::move(p_oclResources));
      }

    NeuralNetwork(Utils::OpenCLResources&& p_oclResources, const H5::H5File& p_file, const size_t p_batchSize)
    : m_batchSize(p_batchSize) {
        m_oclResources = std::make_unique<Utils::OpenCLResources>(std::move(p_oclResources));
        m_inputDimensions = Utils::Dimensions(Utils::readVectorFromHDF5<size_t>(p_file, "inputDimensions"));
    }


    const std::vector<float> getLayersSerializedArgs() const { 
        std::vector<float> layersArgs = {};
        for (const auto& layer : m_layers) {
            auto& layerArgs = layer->getSerializedArgs();
            layersArgs.insert(layersArgs.end(), layerArgs.begin(), layerArgs.end());
        }
        return layersArgs;
    }

    virtual void setBatchSize(const size_t p_batchSize) = 0;

    virtual Utils::NetworkType getType() const = 0;
protected:
    std::vector<std::unique_ptr<Layers::Layer>> m_layers;
    std::unique_ptr<Utils::OpenCLResources> m_oclResources;
    std::unique_ptr<LossFunctions::LossFunction> m_lossFunction;
    
    size_t m_batchSize;
    Utils::Dimensions m_inputDimensions;
};