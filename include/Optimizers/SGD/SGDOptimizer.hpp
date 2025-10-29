#pragma once

#include "Optimizers/Optimizer.hpp"

class SGDOptimizer : public Optimizer {
public:

    SGDOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                 float p_learningRate,
                 float p_weightDecayRate)
        : Optimizer(p_sharedResources, p_learningRate, p_weightDecayRate) { setupKernels(); }

    SGDOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                 const H5::Group& p_optimizerGroup)
        : Optimizer(p_sharedResources, p_optimizerGroup) { setupKernels(); }

    ~SGDOptimizer() = default;

    cl::Event updateParameters(const cl::CommandQueue& p_concurrentQueue, 
                               const cl::Event& p_lastEvent, 
                               const std::string& p_parametersId, 
                               cl::Buffer& p_parameters, 
                               cl::Buffer& p_gradients, 
                               size_t p_numElements) final override;

    Utils::OptimizerType getType() const final override { return Utils::OptimizerType::SGD; }

private:
    void setupKernels() final override;

    void saveSGDOptimizer(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup) const { saveOptimizer(p_queue, p_optimizerGroup); }
    bool sgdOptimizerEquals(const cl::CommandQueue& p_queue, const Optimizer& p_other) const { return optimizerEquals(p_queue, p_other); }
    void printSGDOptimizer() const { printOptimizer(); }
};