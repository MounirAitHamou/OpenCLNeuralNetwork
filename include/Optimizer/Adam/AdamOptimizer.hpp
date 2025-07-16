#pragma once

#include "Optimizer/Optimizer.hpp"
#include "Utils/OpenCLSetup.hpp"

#include <string>
#include <iostream>

class AdamOptimizer : public Optimizer {
public:

    float beta1;
    float beta2;
    float epsilon;
    unsigned int t;

    std::map<std::string, std::pair<cl::Buffer, cl::Buffer>> moment_buffers;

    AdamOptimizer(const OpenCLSetup& ocl_setup,
                  float learning_rate,
                  float weight_decay_rate,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float epsilon = 1e-8f)
        : Optimizer(ocl_setup, learning_rate, weight_decay_rate),
          beta1(beta1), beta2(beta2), epsilon(epsilon), t(1) {}

    AdamOptimizer(const OpenCLSetup& ocl_setup, const H5::Group& optimizer_group);
    ~AdamOptimizer() = default;

    void updateParameters(std::string param_id,
                          cl::Buffer& params_buf,
                          cl::Buffer& grads_buf,
                          size_t num_elements) override;

    void step() override;

    void print() const override {
        std::cout << "Adam Optimizer:\n"
                  << "Learning Rate: " << learning_rate << "\n"
                  << "Weight Decay Rate: " << weight_decay_rate << "\n"
                  << "Beta1: " << beta1 << "\n"
                  << "Beta2: " << beta2 << "\n"
                  << "Epsilon: " << epsilon << "\n";
    }

    OptimizerType getType() const override {
        return OptimizerType::Adam;
    }

    void saveOptimizer(H5::Group& optimizer_group,
                                  const std::map<size_t, std::pair<size_t, size_t>>& moments_sizes) const override;
};