#pragma once
#include "LossFunctions/AllLossFunctions.hpp"
#include <memory>
#include <stdexcept>
#include <vector>
#include <iterator>

namespace Utils {
    struct LossFunctionArgs {
        virtual ~LossFunctionArgs() = default;
        virtual LossFunctionType getLossFunctionType() const = 0;
        virtual std::unique_ptr<LossFunctions::LossFunction> createLossFunction(
            std::shared_ptr<Utils::SharedResources> p_sharedResources
        ) const = 0;
    };

    struct MeanSquaredErrorLossFunctionArgs : public LossFunctionArgs {
    public:
        MeanSquaredErrorLossFunctionArgs() = default;

        std::unique_ptr<LossFunctions::LossFunction> createLossFunction(std::shared_ptr<Utils::SharedResources> p_sharedResources) const final override {
            return std::make_unique<LossFunctions::MeanSquaredError>(p_sharedResources);
        }

        LossFunctionType getLossFunctionType() const override {
            return LossFunctionType::MeanSquaredError;
        }
    };

    struct CategoricalCrossEntropyLossFunctionArgs : public LossFunctionArgs {
    public:
        CategoricalCrossEntropyLossFunctionArgs() = default;

        std::unique_ptr<LossFunctions::LossFunction> createLossFunction(std::shared_ptr<Utils::SharedResources> p_sharedResources) const final override {
            return std::make_unique<LossFunctions::CategoricalCrossEntropy>(p_sharedResources);
        }

        LossFunctionType getLossFunctionType() const override {
            return LossFunctionType::CategoricalCrossEntropy;
        }
    };

    struct BinaryCrossEntropyLossFunctionArgs : public LossFunctionArgs {
    public:
        BinaryCrossEntropyLossFunctionArgs() = default;

        std::unique_ptr<LossFunctions::LossFunction> createLossFunction(std::shared_ptr<Utils::SharedResources> p_sharedResources) const final override {
            return std::make_unique<LossFunctions::BinaryCrossEntropy>(p_sharedResources);
        }

        LossFunctionType getLossFunctionType() const override {
            return LossFunctionType::BinaryCrossEntropy;
        }
    };

    struct SoftmaxCrossEntropyLossFunctionArgs : public LossFunctionArgs {
    public:
        SoftmaxCrossEntropyLossFunctionArgs() = default;

        std::unique_ptr<LossFunctions::LossFunction> createLossFunction(std::shared_ptr<Utils::SharedResources> p_sharedResources) const final override {
            return std::make_unique<LossFunctions::SoftmaxCrossEntropy>(p_sharedResources);
        }

        LossFunctionType getLossFunctionType() const override {
            return LossFunctionType::SoftmaxCrossEntropy;
        }
    };

    std::unique_ptr<LossFunctionArgs> makeMeanSquaredErrorLossFunctionArgs();
    std::unique_ptr<LossFunctionArgs> makeCategoricalCrossEntropyLossFunctionArgs();
    std::unique_ptr<LossFunctionArgs> makeBinaryCrossEntropyLossFunctionArgs();
    std::unique_ptr<LossFunctionArgs> makeSoftmaxCrossEntropyLossFunctionArgs();

    std::unique_ptr<LossFunctions::LossFunction> loadLossFunction(std::shared_ptr<Utils::SharedResources> p_sharedResources, const H5::Group& p_lossFunctionGroup);
}