#pragma once

#include "Optimizer/Optimizer.hpp"
#include "Utils/OpenCLSetup.hpp"

#include <iostream>

class SGDOptimizer : public Optimizer {
public:

    SGDOptimizer(const OpenCLSetup& ocl_setup,
                 float learning_rate,
                 float weight_decay_rate)
        : Optimizer(ocl_setup, learning_rate, weight_decay_rate) {}

    SGDOptimizer(const OpenCLSetup& ocl_setup, const H5::Group& optimizer_group);

    ~SGDOptimizer() = default;

    void updateParameters(std::string param_id,
                          cl::Buffer& params_buf,
                          cl::Buffer& grads_buf,
                          size_t num_elements) override;

    void print() const override {
        std::cout << "SGD Optimizer:\n"
                  << "Learning Rate: " << learning_rate << "\n"
                  << "Weight Decay Rate: " << weight_decay_rate << "\n";
    }

    OptimizerType getType() const override {
        return OptimizerType::SGD;
    }

    void saveOptimizer(H5::Group& optimizer_group,
                                  const std::map<size_t, std::pair<size_t, size_t>>& moments_sizes) const override;
};