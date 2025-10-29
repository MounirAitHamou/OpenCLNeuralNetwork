#pragma once

#include "Optimizers/AdamBaseOptimizers/AdamBaseOptimizer.hpp"

class AdamWOptimizer : public AdamBaseOptimizer {
public:
    AdamWOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                  float p_learningRate,
                  float p_weightDecayRate,
                  float p_beta1 = 0.9f,
                  float p_beta2 = 0.999f,
                  float p_epsilon = 1e-8f)
        : AdamBaseOptimizer(p_sharedResources, p_learningRate, p_weightDecayRate, p_beta1, p_beta2, p_epsilon) { setupKernels(); }

    AdamWOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                  const H5::Group& p_optimizerGroup)
        : AdamBaseOptimizer(p_sharedResources, p_optimizerGroup) { setupKernels(); }

    ~AdamWOptimizer() = default;

    Utils::OptimizerType getType() const final override { return  Utils::OptimizerType::AdamW; }


protected:
        void setupKernels() final override;

        void saveAdamWOptimizer(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup, const std::map<size_t, std::pair<size_t, size_t>>& p_momentSizes) const { saveAdamBaseOptimizer(p_queue, p_optimizerGroup, p_momentSizes); }
        bool adamWOptimizerEquals(const cl::CommandQueue& p_queue, const Optimizer& p_other, std::map<size_t, std::pair<size_t, size_t>>& p_momentSizes) const {return adamBaseOptimizerEquals(p_queue, p_other, p_momentSizes); }
        void printAdamWOptimizer() const { printAdamBaseOptimizer(); }
};