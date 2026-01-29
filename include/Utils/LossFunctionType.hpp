#pragma once

#include <stdexcept>
namespace Utils {
    enum class LossFunctionType : unsigned int {
        MeanSquaredError = 0,
        BinaryCrossEntropy = 1,
        CategoricalCrossEntropy = 2,
        SoftmaxCrossEntropy = 3
    };

    inline LossFunctionType lossFunctionTypeFromUint(unsigned int p_val) {
        switch(p_val) {
            case 0: return LossFunctionType::MeanSquaredError;
            case 1: return LossFunctionType::BinaryCrossEntropy;
            case 2: return LossFunctionType::CategoricalCrossEntropy;
            case 3: return LossFunctionType::SoftmaxCrossEntropy;
            default:
                throw std::invalid_argument("Invalid value for LossFunctionType");
        }
    }

    inline std::string lossFunctionTypeToString(LossFunctionType p_type) {
        switch(p_type) {
            case LossFunctionType::MeanSquaredError: return "MeanSquaredError";
            case LossFunctionType::BinaryCrossEntropy: return "BinaryCrossEntropy";
            case LossFunctionType::CategoricalCrossEntropy: return "CategoricalCrossEntropy";
            case LossFunctionType::SoftmaxCrossEntropy: return "SoftmaxCrossEntropy";
            default:
                return "Unknown";
        }
    }
}
