#pragma once

#include <stdexcept>
namespace Utils {
    enum class LossFunctionType : unsigned int {
        MeanSquaredError = 0,
        BinaryCrossEntropy = 1,
    };

    inline LossFunctionType lossFunctionTypeFromUint(unsigned int p_val) {
        switch(p_val) {
            case 0: return LossFunctionType::MeanSquaredError;
            case 1: return LossFunctionType::BinaryCrossEntropy;
            default:
                throw std::invalid_argument("Invalid value for LossFunctionType");
        }
    }

    inline double applyLossFunction(LossFunctionType p_lossFunction, float p_prediction, float p_target) {
        switch (p_lossFunction) {
            case LossFunctionType::MeanSquaredError:
                return (p_prediction - p_target) * (p_prediction - p_target);
            case LossFunctionType::BinaryCrossEntropy:
                return - (p_target * std::log(p_prediction + 1e-17f) + (1 - p_target) * std::log(1 - p_prediction + 1e-17f));
            default:
                throw std::invalid_argument("Unknown LossFunctionType");
        }
    }
}
