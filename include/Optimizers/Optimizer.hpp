#pragma once

#include "Utils/OpenCLResources.hpp"
#include "Utils/OptimizerType.hpp"
#include "Layers/TrainableLayers/TrainableLayer.hpp"
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <utility>
namespace Optimizers {
    class Optimizer {
    public:
        Optimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                float p_learningRate,
                float p_weightDecayRate)
                : m_sharedResources(p_sharedResources),
                    m_learningRate(p_learningRate),
                    m_weightDecayRate(p_weightDecayRate) {}

        Optimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                const H5::Group& p_optimizerGroup)
            : m_sharedResources(p_sharedResources) {
                m_learningRate = Utils::readValueFromHDF5<float>(p_optimizerGroup, "learningRate");
                m_weightDecayRate = Utils::readValueFromHDF5<float>(p_optimizerGroup, "weightDecayRate");
            }

        virtual ~Optimizer() = default;

        std::pair<cl::Event, cl::Event> updateTrainableLayer(
            const cl::CommandQueue& p_concurrentQueue,
            const std::pair<cl::Event, cl::Event>& p_prevEvents,
            Layers::Trainable::TrainableLayer& p_layer)
        {
            const size_t layerId = p_layer.getLayerId();
            cl::Event weightEvent = updateParameters(
                p_concurrentQueue,
                p_prevEvents.first,
                std::to_string(layerId) + "Weights",
                p_layer.getWeights(),
                p_layer.getWeightsGradients(),
                p_layer.getWeightsSize()
            );

            cl::Event biasEvent = updateParameters(
                p_concurrentQueue,
                p_prevEvents.second,
                std::to_string(layerId) + "Biases",
                p_layer.getBiases(),
                p_layer.getBiasesGradients(),
                p_layer.getBiasesSize()
            );

            return { weightEvent, biasEvent };
        }

        virtual cl::Event updateParameters(const cl::CommandQueue& p_concurrentQueue, 
                                        const cl::Event& p_lastEvent, 
                                        const std::string& p_parametersId, 
                                        cl::Buffer& p_parameters, 
                                        cl::Buffer& p_gradients, 
                                        size_t p_numElements) = 0;

        virtual Utils::OptimizerType getType() const = 0;

        virtual void save(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup, const std::map<size_t, std::pair<size_t, size_t>>& p_parameterSizes) const { saveOptimizer(p_queue, p_optimizerGroup); }
        virtual bool equals(const cl::CommandQueue& p_queue, const Optimizer& p_other, std::map<size_t, std::pair<size_t, size_t>>& p_parameterSizes) const { return optimizerEquals(p_queue, p_other); }
        virtual void print() const { printOptimizer(); }
        
        virtual void step() {}
    protected:
        float m_learningRate;
        float m_weightDecayRate;

        std::shared_ptr<Utils::SharedResources> m_sharedResources;

        cl::Kernel m_updateKernel;

        virtual void setupKernels() = 0;

        void saveOptimizer(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup) const {
            unsigned int optimizerType = static_cast<unsigned int>(getType());
            Utils::writeValueToHDF5<unsigned int>(p_optimizerGroup, "optimizerType", optimizerType);
            Utils::writeValueToHDF5<float>(p_optimizerGroup, "learningRate", m_learningRate);
            Utils::writeValueToHDF5<float>(p_optimizerGroup, "weightDecayRate", m_weightDecayRate);
        }

        bool optimizerEquals(const cl::CommandQueue& p_queue, const Optimizer& p_other) const {
            return getType() == p_other.getType() && 
                m_learningRate == p_other.m_learningRate && 
                m_weightDecayRate == p_other.m_weightDecayRate;
        }

        void printOptimizer() const {
            std::cout << "Optimizer Type: " << Utils::optimizerTypeToString(getType()) << "\n"
                    << "Learning Rate: " << m_learningRate << "\n"
                    << "Weight Decay Rate: " << m_weightDecayRate << "\n";
        }
    };
}