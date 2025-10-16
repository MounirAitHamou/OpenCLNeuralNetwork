#pragma once

#include <stdexcept>
namespace Utils {

    enum class PaddingType : unsigned int {
        Valid = 0,
        Same = 1,
    };

    inline PaddingType paddingTypeFromUint(unsigned int p_val) {
        switch(p_val) {
            case 0: return PaddingType::Valid;
            case 1: return PaddingType::Same;
            default:
                throw std::invalid_argument("Invalid value for PaddingType");
        }
    }
}
