#pragma once

#include <stdexcept>

enum class LayerType : unsigned int {
    Dense = 0,
};

inline LayerType layerTypeFromUint(unsigned int val) {
    switch(val) {
        case 0: return LayerType::Dense;
        // Add other cases for new layer types here
        default:
            throw std::invalid_argument("Invalid value for LayerType");
    }
}