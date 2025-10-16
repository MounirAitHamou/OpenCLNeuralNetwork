#pragma once

#include "Optimizer/Optimizer.hpp"

class AdamWOptimizer : public Optimizer {
public:
    AdamWOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                   float p_learningRate,
                   float p_weightDecayRate,
                   float p_beta1 = 0.9f,
                   float p_beta2 = 0.999f,
                   float p_epsilon = 1e-8f);

    AdamWOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                   const H5::Group& p_optimizerGroup);
                   
    ~AdamWOptimizer() = default;

    bool equals(const cl::CommandQueue& p_queue, const Optimizer& p_other, std::map<size_t, std::pair<size_t, size_t>>& p_momentsSizes) const override;

    cl::Event updateParameters(const cl::CommandQueue& p_concurrentQueue, 
                               cl::Event& p_lastEvent, 
                               const std::string& p_parametersId, 
                               cl::Buffer& p_parameters, 
                               cl::Buffer& p_gradients, 
                               size_t p_numElements) override;

    void step() override;

    void print() const override {
        std::cout << "AdamW Optimizer:\n"
                  << "Learning Rate: " << m_learningRate << "\n"
                  << "Weight Decay Rate: " << m_weightDecayRate << "\n"
                  << "Beta1: " << m_beta1 << "\n"
                  << "Beta2: " << m_beta2 << "\n"
                  << "Epsilon: " << m_epsilon << "\n";
    }

    Utils::OptimizerType getType() const override {
        return Utils::OptimizerType::AdamW;
    }

    void saveOptimizer(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup,
        const std::map<size_t, std::pair<size_t, size_t>>& p_momentsSizes) const override;

private:
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    unsigned int m_t;

    std::map<std::string, std::pair<cl::Buffer, cl::Buffer>> m_momentBuffers;
    void setupKernels() override;
};