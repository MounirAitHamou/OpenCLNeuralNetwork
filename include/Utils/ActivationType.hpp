#pragma once

#include <stdexcept>
namespace Utils {
    
    enum class ActivationType : unsigned int {
        Linear = 0,
        ReLU = 1,
        Sigmoid = 2,
        Tanh = 3,
    };

    inline ActivationType activationTypeFromUint(unsigned int p_val) {
        switch(p_val) {
            case 0: return ActivationType::Linear;
            case 1: return ActivationType::ReLU;
            case 2: return ActivationType::Sigmoid;
            case 3: return ActivationType::Tanh;
            default:
                throw std::invalid_argument("Invalid value for ActivationType");
        }
    }
}