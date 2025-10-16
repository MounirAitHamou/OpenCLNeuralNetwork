#pragma once

namespace Utils {
    struct PaddingValues : public Dimensions {

        PaddingValues(std::vector<size_t> p_dimensions) {
            if (p_dimensions.size() == 4) {
                for (const auto& dim : p_dimensions) {
                    if (dim < 0) {
                        throw std::invalid_argument("Dimensions cannot be negative.");
                    }
                }
                m_dimensions = p_dimensions;
            }
            else {
                throw std::invalid_argument("Padding Values require a 4-dimensional vector.");
            }
        }

        PaddingValues(std::initializer_list<size_t> p_dimensions) {
            if (p_dimensions.size() == 4) {
                for (const auto& dim : p_dimensions) {
                    if (dim < 0) {
                        throw std::invalid_argument("Dimensions cannot be negative.");
                    }
                }
                m_dimensions = p_dimensions;
            }
            else {
                throw std::invalid_argument("Padding Values require a 4-dimensional vector.");
            }
        }

        PaddingValues(size_t p_padTop, size_t p_padBottom, size_t p_padLeft, size_t p_padRight)
        {
            if (p_padTop < 0 || p_padBottom < 0 || p_padLeft < 0 || p_padRight < 0){
                throw std::invalid_argument("Padding Values cannot be negative.");
            }
            m_dimensions = {p_padTop, p_padBottom, p_padLeft, p_padRight};
        }
        
        
        PaddingValues() : PaddingValues(1, 1, 1, 1) {}

        size_t getTop() const { return m_dimensions[0]; }
        size_t getBottom() const { return m_dimensions[1]; }
        size_t getLeft() const { return m_dimensions[2]; }
        size_t getRight() const { return m_dimensions[3]; }
    };
}