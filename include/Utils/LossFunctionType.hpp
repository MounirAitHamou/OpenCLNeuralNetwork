#pragma once

#include <stdexcept>

enum class LossFunctionType : unsigned int {
    MeanSquaredError = 0,
    BinaryCrossEntropy = 1,
};

inline LossFunctionType lossFunctionTypeFromUint(unsigned int val) {
    switch(val) {
        case 0: return LossFunctionType::MeanSquaredError;
        case 1: return LossFunctionType::BinaryCrossEntropy;
        default:
            throw std::invalid_argument("Invalid value for LossFunctionType");
    }
}