#pragma once

#include <stdexcept>
#include <string>

namespace Utils {
    enum class LayerType : unsigned int {
        Dense = 0,
        Convolutional = 1,
        ReLU = 2,
        LeakyReLU = 3,
        Sigmoid = 4,
        Softmax = 5,
        Tanh = 6
    };

    inline LayerType layerTypeFromUint(unsigned int p_val) {
        switch(p_val) {
            case 0: return LayerType::Dense;
            case 1: return LayerType::Convolutional;
            case 2: return LayerType::ReLU;
            case 3: return LayerType::LeakyReLU;
            case 4: return LayerType::Sigmoid;
            case 5: return LayerType::Softmax;
            case 6: return LayerType::Tanh;
            default:
                throw std::invalid_argument("Invalid value for LayerType");
        }
    }

    inline std::string layerTypeToString(LayerType p_type) {
        switch(p_type) {
            case LayerType::Dense: return "Dense";
            case LayerType::Convolutional: return "Convolutional";
            case LayerType::ReLU: return "ReLU";
            case LayerType::LeakyReLU: return "LeakyReLU";
            case LayerType::Sigmoid: return "Sigmoid";
            case LayerType::Softmax: return "Softmax";
            case LayerType::Tanh: return "Tanh";
            default:
                throw std::invalid_argument("Invalid LayerType value");
        }
    }
}
