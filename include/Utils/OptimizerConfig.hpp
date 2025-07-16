#pragma once

#include "Optimizer/AllOptimizers.hpp"
#include "Utils/OpenCLSetup.hpp"
#include <memory>
#include <iostream>


namespace OptimizerConfig {

    struct OptimizerParameters {

        float learning_rate;

        float weight_decay_rate;

        OptimizerParameters(float lr = 0.01f, float wd = 0.0f)
            : learning_rate(lr), weight_decay_rate(wd) {}

        virtual ~OptimizerParameters() = default;

        virtual std::unique_ptr<Optimizer> createOptimizer(const OpenCLSetup& ocl_setup) const = 0;

        virtual OptimizerType getOptimizerType() const = 0;

        virtual void print() const = 0;

        virtual std::unique_ptr<OptimizerParameters> clone() const = 0;
    };

    struct SGDOptimizerParameters : public OptimizerParameters {

        SGDOptimizerParameters(float lr = 0.01f, float wd = 0.0f)
            : OptimizerParameters(lr, wd) {}

        std::unique_ptr<Optimizer> createOptimizer(const OpenCLSetup& ocl_setup) const override {
            return std::make_unique<SGDOptimizer>(ocl_setup, learning_rate, weight_decay_rate);
        }

        OptimizerType getOptimizerType() const override {
            return OptimizerType::SGD;
        }

        void print() const override {
            std::cout << "SGD Optimizer Parameters:\n"
                      << "Learning Rate: " << learning_rate << "\n"
                      << "Weight Decay Rate: " << weight_decay_rate << "\n";
        }

        std::unique_ptr<OptimizerParameters> clone() const override {
            return std::make_unique<SGDOptimizerParameters>(*this);
        }
    };

    struct AdamOptimizerParameters : public OptimizerParameters {

        float beta1;

        float beta2;

        float epsilon;

        AdamOptimizerParameters(float lr = 0.01f, float wd = 0.0f,
                                float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
            : OptimizerParameters(lr, wd), beta1(b1), beta2(b2), epsilon(eps) {}

        std::unique_ptr<Optimizer> createOptimizer(const OpenCLSetup& ocl_setup) const override {
            return std::make_unique<AdamOptimizer>(ocl_setup, learning_rate, weight_decay_rate, beta1, beta2, epsilon);
        }

        OptimizerType getOptimizerType() const override {
            return OptimizerType::Adam;
        }

        void print() const override {
            std::cout << "Adam Optimizer Parameters:\n"
                      << "Learning Rate: " << learning_rate << "\n"
                      << "Weight Decay Rate: " << weight_decay_rate << "\n"
                      << "Beta1: " << beta1 << "\n"
                      << "Beta2: " << beta2 << "\n"
                      << "Epsilon: " << epsilon << "\n";
        }

        std::unique_ptr<OptimizerParameters> clone() const override {
            return std::make_unique<AdamOptimizerParameters>(*this);
        }
    };

    struct AdamWOptimizerParameters : public OptimizerParameters {

        float beta1;

        float beta2;

        float epsilon;

        AdamWOptimizerParameters(float lr = 0.01f, float wd = 0.0f,
                                 float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
            : OptimizerParameters(lr, wd), beta1(b1), beta2(b2), epsilon(eps) {}

        std::unique_ptr<Optimizer> createOptimizer(const OpenCLSetup& ocl_setup) const override {
            return std::make_unique<AdamWOptimizer>(ocl_setup, learning_rate, weight_decay_rate, beta1, beta2, epsilon);
        }

        OptimizerType getOptimizerType() const override {
            return OptimizerType::AdamW;
        }

        void print() const override {
            std::cout << "AdamW Optimizer Parameters:\n"
                      << "Learning Rate: " << learning_rate << "\n"
                      << "Weight Decay Rate: " << weight_decay_rate << "\n"
                      << "Beta1: " << beta1 << "\n"
                      << "Beta2: " << beta2 << "\n"
                      << "Epsilon: " << epsilon << "\n";
        }

        std::unique_ptr<OptimizerParameters> clone() const override {
            return std::make_unique<AdamWOptimizerParameters>(*this);
        }
    };


    SGDOptimizerParameters makeSGDParameters(float learning_rate = 0.01f, float weight_decay_rate = 0.0f);

    AdamOptimizerParameters makeAdamParameters(float learning_rate = 0.01f, float weight_decay_rate = 0.0f,
                                               float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    AdamWOptimizerParameters makeAdamWParameters(float learning_rate = 0.01f, float weight_decay_rate = 0.0f,
                                                 float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    std::unique_ptr<Optimizer> makeOptimizer(const OpenCLSetup& ocl_setup, const OptimizerParameters& params);

    std::unique_ptr<Optimizer> loadOptimizer(const OpenCLSetup& ocl_setup, const H5::Group& optimizer_group);
}