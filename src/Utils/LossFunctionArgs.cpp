#include "Utils/LossFunctionArgs.hpp"

namespace Utils {
    std::unique_ptr<LossFunctionArgs> makeMeanSquaredErrorLossFunctionArgs() {
        return std::make_unique<MeanSquaredErrorLossFunctionArgs>();
    }

    std::unique_ptr<LossFunctionArgs> makeCategoricalCrossEntropyLossFunctionArgs() {
        return std::make_unique<CategoricalCrossEntropyLossFunctionArgs>();
    }

    std::unique_ptr<LossFunctionArgs> makeBinaryCrossEntropyLossFunctionArgs() {
        return std::make_unique<BinaryCrossEntropyLossFunctionArgs>();
    }

    std::unique_ptr<LossFunctionArgs> makeSoftmaxCrossEntropyLossFunctionArgs() {
        return std::make_unique<SoftmaxCrossEntropyLossFunctionArgs>();
    }

    std::unique_ptr<LossFunctions::LossFunction> loadLossFunction(std::shared_ptr<Utils::SharedResources> p_sharedResources, const H5::Group& p_lossFunctionGroup) {
        unsigned int lossFunctionType;
        p_lossFunctionGroup.openAttribute("lossFunctionType").read(H5::PredType::NATIVE_UINT, &lossFunctionType);

        switch (lossFunctionTypeFromUint(lossFunctionType)) {
            case LossFunctionType::MeanSquaredError:
                return std::make_unique<LossFunctions::MeanSquaredError>(p_sharedResources);
            case LossFunctionType::BinaryCrossEntropy:
                return std::make_unique<LossFunctions::BinaryCrossEntropy>(p_sharedResources);
            case LossFunctionType::CategoricalCrossEntropy:
                return std::make_unique<LossFunctions::CategoricalCrossEntropy>(p_sharedResources);
            case LossFunctionType::SoftmaxCrossEntropy:
                return std::make_unique<LossFunctions::SoftmaxCrossEntropy>(p_sharedResources);
            default:
                throw std::invalid_argument("Invalid LossFunctionType in HDF5 group");
        }
    }
}