#include "Utils/Dimensions.hpp"

size_t Dimensions::getTotalElements() const {
    if (dims.empty()) {
        return 0;
    }
    return std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>());
}