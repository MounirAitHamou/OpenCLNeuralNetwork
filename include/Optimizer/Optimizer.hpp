#pragma once

#include "Utils/OpenCLResources.hpp"
#include "Utils/OptimizerType.hpp"
#include <string>
#include <iostream>
#include <vector>
#include <map>

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
            p_optimizerGroup.openAttribute("learningRate").read(H5::PredType::NATIVE_FLOAT, &m_learningRate);
            p_optimizerGroup.openAttribute("weightDecayRate").read(H5::PredType::NATIVE_FLOAT, &m_weightDecayRate);
        }

    virtual ~Optimizer() = default;

    virtual bool equals(const cl::CommandQueue& p_queue, const Optimizer& p_other, std::map<size_t, std::pair<size_t, size_t>>& p_momentsSizes) const = 0;

    virtual cl::Event updateParameters(const cl::CommandQueue& p_concurrentQueue, 
                                       cl::Event& p_lastEvent, 
                                       const std::string& p_parametersId, 
                                       cl::Buffer& p_parameters, 
                                       cl::Buffer& p_gradients, 
                                       size_t p_numElements) = 0;

    virtual void step() {}

    virtual void print() const = 0;
    
    virtual Utils::OptimizerType getType() const = 0;

    virtual void saveOptimizer(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup,
        const std::map<size_t, std::pair<size_t, size_t>>& p_momentsSizes) const = 0;

protected:
    float m_learningRate;
    float m_weightDecayRate;

    std::shared_ptr<Utils::SharedResources> m_sharedResources;

    cl::Kernel m_updateKernel;
    virtual void setupKernels() = 0;

    bool compareBuffers(const cl::CommandQueue& p_queue, const std::string& p_key, size_t p_size,
                        const std::map<std::string, std::pair<cl::Buffer, cl::Buffer>>& p_buffers1, 
                        const std::map<std::string, std::pair<cl::Buffer, cl::Buffer>>& p_buffers2) const {
        const float epsilon = 1e-6f;
        if (p_size == 0) {
            return true;
        }
        
        auto it1 = p_buffers1.find(p_key);
        auto it2 = p_buffers2.find(p_key);

        if ((it1 == p_buffers1.end() && it2 != p_buffers2.end()) || (it1 != p_buffers1.end() && it2 == p_buffers2.end())) {
            return false;
        }

        std::vector<float> data1 = Utils::readCLBuffer(p_queue, it1->second.first, p_size);
        std::vector<float> data2 = Utils::readCLBuffer(p_queue, it2->second.first, p_size);
        
        for (size_t i = 0; i < p_size; ++i) {
            if (fabs(data1[i] - data2[i]) > epsilon) {
                return false;
            }
        }

        std::vector<float> data1_v = Utils::readCLBuffer(p_queue, it1->second.second, p_size);
        std::vector<float> data2_v = Utils::readCLBuffer(p_queue, it2->second.second, p_size);
        for (size_t i = 0; i < p_size; ++i) {
            if (fabs(data1_v[i] - data2_v[i]) > epsilon) {
                return false;
            }
        }
        
        return true;
    }
};