#pragma once

#include <stdexcept>
#include <string>
namespace Utils {
    enum class OptimizerType : unsigned int {
        SGD = 0,
        Adam = 1,
        AdamW = 2,
    };

    inline OptimizerType optimizerTypeFromUint(unsigned int p_val) {
        switch(p_val) {
            case 0: return OptimizerType::SGD;
            case 1: return OptimizerType::Adam;
            case 2: return OptimizerType::AdamW;
            default:
                throw std::invalid_argument("Invalid value for OptimizerType");
        }
    }

    inline std::string optimizerTypeToString(OptimizerType p_type) {
        switch(p_type) {
            case OptimizerType::SGD: return "SGD";
            case OptimizerType::Adam: return "Adam";
            case OptimizerType::AdamW: return "AdamW";
            default:
                return "Unknown";
        }
    }
}