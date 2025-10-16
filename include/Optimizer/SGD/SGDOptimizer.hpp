#pragma once

#include "Optimizer/Optimizer.hpp"

class SGDOptimizer : public Optimizer {
public:

    SGDOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                 float p_learningRate,
                 float p_weightDecayRate);

    SGDOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                 const H5::Group& p_optimizerGroup);

    ~SGDOptimizer() = default;

    bool equals(const cl::CommandQueue& p_queue, const Optimizer& p_other, std::map<size_t, std::pair<size_t, size_t>>& p_momentsSizes) const override;

    cl::Event updateParameters(const cl::CommandQueue& p_concurrentQueue, 
                               cl::Event& p_lastEvent, 
                               const std::string& p_parametersId, 
                               cl::Buffer& p_parameters, 
                               cl::Buffer& p_gradients, 
                               size_t p_numElements) override;

    void print() const override {
        std::cout << "SGD Optimizer:\n"
                  << "Learning Rate: " << m_learningRate << "\n"
                  << "Weight Decay Rate: " << m_weightDecayRate << "\n";
    }

    Utils::OptimizerType getType() const override {
        return Utils::OptimizerType::SGD;
    }

    void saveOptimizer(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup,
        const std::map<size_t, std::pair<size_t, size_t>>& p_momentsSizes) const override;

private:
    void setupKernels() override;
};