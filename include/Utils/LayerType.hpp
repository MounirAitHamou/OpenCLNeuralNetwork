#pragma once

#include <stdexcept>

namespace Utils {
    enum class LayerType : unsigned int {
        Dense = 0,
        Convolutional = 1,
    };

    inline LayerType layerTypeFromUint(unsigned int p_val) {
        switch(p_val) {
            case 0: return LayerType::Dense;
            case 1: return LayerType::Convolutional;
            default:
                throw std::invalid_argument("Invalid value for LayerType");
        }
    }
}
