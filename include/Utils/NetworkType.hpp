#pragma once

#include <stdexcept>
#include <string>

namespace Utils {
    enum class NetworkType : unsigned int {
        Local = 0,
    };

    inline NetworkType networkTypeFromUint(unsigned int p_val) {
        switch(p_val) {
            case 0: return NetworkType::Local;
            default:
                throw std::invalid_argument("Invalid value for NetworkType");
        }
    }

    inline std::string networkTypeToString(NetworkType p_type) {
        switch(p_type) {
            case NetworkType::Local: return "Local";
            default:
                throw std::invalid_argument("Invalid NetworkType value");
        }
    }
}