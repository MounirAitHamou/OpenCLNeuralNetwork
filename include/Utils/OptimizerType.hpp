#pragma once

#include <stdexcept>

enum class OptimizerType : unsigned int {
    SGD = 0,
    Adam = 1,
    AdamW = 2,
};

inline OptimizerType optimizerTypeFromUint(unsigned int val) {
    switch(val) {
        case 0: return OptimizerType::SGD;
        case 1: return OptimizerType::Adam;
        case 2: return OptimizerType::AdamW;
        default:
            throw std::invalid_argument("Invalid value for OptimizerType");
    }
}