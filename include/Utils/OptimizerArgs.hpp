#pragma once

#include "Optimizer/AllOptimizers.hpp"
#include "Utils/OpenCLResources.hpp"
#include <memory>
#include <iostream>

namespace Utils {
    struct OptimizerArgs {
    protected:
        float m_learningRate;
        float m_weightDecayRate;
    public:
        OptimizerArgs(float p_learningRate = 0.01f, float p_weightDecayRate = 0.0f)
            : m_learningRate(p_learningRate), m_weightDecayRate(p_weightDecayRate) {}

        virtual ~OptimizerArgs() = default;

        virtual std::unique_ptr<Optimizer> createOptimizer(std::shared_ptr<SharedResources> p_sharedResources) const = 0;

        virtual OptimizerType getOptimizerType() const = 0;

        virtual void print() const = 0;

        virtual std::unique_ptr<OptimizerArgs> clone() const = 0;

        const float getLearningRate() const {
            return m_learningRate;
        }

        const float getWeightDecayRate() const {
            return m_weightDecayRate;
        }
    };

    struct SGDOptimizerArgs : public OptimizerArgs {
    public:
        SGDOptimizerArgs(float p_learningRate = 0.01f, float p_weightDecayRate = 0.0f)
            : OptimizerArgs(p_learningRate, p_weightDecayRate) {}

        std::unique_ptr<Optimizer> createOptimizer(std::shared_ptr<SharedResources> p_sharedResources) const override {
            return std::make_unique<SGDOptimizer>(p_sharedResources, m_learningRate, m_weightDecayRate);
        }

        OptimizerType getOptimizerType() const override {
            return OptimizerType::SGD;
        }

        void print() const override {
            std::cout << "SGD Optimizer Arguments:\n"
                    << "Learning Rate: " << m_learningRate << "\n"
                    << "Weight Decay Rate: " << m_weightDecayRate << "\n";
        }

        std::unique_ptr<OptimizerArgs> clone() const override {
            return std::make_unique<SGDOptimizerArgs>(*this);
        }
    };

    struct AdamOptimizerArgs : public OptimizerArgs {
    private:
        float m_beta1;
        float m_beta2;
        float m_epsilon;
    
    public:
        AdamOptimizerArgs(float p_learningRate = 0.01f, float p_weightDecayRate = 0.0f,
                                float p_beta1 = 0.9f, float p_beta2 = 0.999f, float p_epsilon = 1e-8f)
            : OptimizerArgs(p_learningRate, p_weightDecayRate), m_beta1(p_beta1), m_beta2(p_beta2), m_epsilon(p_epsilon) {}

        std::unique_ptr<Optimizer> createOptimizer(std::shared_ptr<SharedResources> p_sharedResources) const override {
            return std::make_unique<AdamOptimizer>(p_sharedResources, m_learningRate, m_weightDecayRate, m_beta1, m_beta2, m_epsilon);
        }

        OptimizerType getOptimizerType() const override {
            return OptimizerType::Adam;
        }

        const float getBeta1() const {
            return m_beta1;
        }

        const float getBeta2() const {
            return m_beta2;
        }

        const float getEpsilon() const {
            return m_epsilon;
        }

        void print() const override {
            std::cout << "Adam Optimizer Arguments:\n"
                    << "Learning Rate: " << m_learningRate << "\n"
                    << "Weight Decay Rate: " << m_weightDecayRate << "\n"
                    << "Beta1: " << m_beta1 << "\n"
                    << "Beta2: " << m_beta2 << "\n"
                    << "Epsilon: " << m_epsilon << "\n";
        }

        std::unique_ptr<OptimizerArgs> clone() const override {
            return std::make_unique<AdamOptimizerArgs>(*this);
        }
    };

    struct AdamWOptimizerArgs : public OptimizerArgs {
    private:
        float m_beta1;
        float m_beta2;
        float m_epsilon;
    
    public:
        AdamWOptimizerArgs(float p_learningRate = 0.01f, float p_weightDecayRate = 0.0f,
                                float p_beta1 = 0.9f, float p_beta2 = 0.999f, float p_epsilon = 1e-8f)
            : OptimizerArgs(p_learningRate, p_weightDecayRate), m_beta1(p_beta1), m_beta2(p_beta2), m_epsilon(p_epsilon) {}

        std::unique_ptr<Optimizer> createOptimizer(std::shared_ptr<SharedResources> p_sharedResources) const override {
            return std::make_unique<AdamWOptimizer>(p_sharedResources, m_learningRate, m_weightDecayRate, m_beta1, m_beta2, m_epsilon);
        }

        OptimizerType getOptimizerType() const override {
            return OptimizerType::AdamW;
        }

        const float getBeta1() const {
            return m_beta1;
        }

        const float getBeta2() const {
            return m_beta2;
        }

        const float getEpsilon() const {
            return m_epsilon;
        }

        void print() const override {
            std::cout << "AdamW Optimizer Arguments:\n"
                    << "Learning Rate: " << m_learningRate << "\n"
                    << "Weight Decay Rate: " << m_weightDecayRate << "\n"
                    << "Beta1: " << m_beta1 << "\n"
                    << "Beta2: " << m_beta2 << "\n"
                    << "Epsilon: " << m_epsilon << "\n";
        }

        std::unique_ptr<OptimizerArgs> clone() const override {
            return std::make_unique<AdamWOptimizerArgs>(*this);
        }
    };


    SGDOptimizerArgs makeSGDArgs(float p_learningRate = 0.01f, float p_weightDecayRate = 0.0f);

    AdamOptimizerArgs makeAdamArgs(float p_learningRate = 0.01f, float p_weightDecayRate = 0.0f,
                                               float p_beta1 = 0.9f, float p_beta2 = 0.999f, float p_epsilon = 1e-8f);

    AdamWOptimizerArgs makeAdamWArgs(float p_learningRate = 0.01f, float p_weightDecayRate = 0.0f,
                                               float p_beta1 = 0.9f, float p_beta2 = 0.999f, float p_epsilon = 1e-8f);

    std::unique_ptr<Optimizer> loadOptimizer(
        std::shared_ptr<SharedResources> p_sharedResources,
        const H5::Group& p_optimizerGroup
    );
}