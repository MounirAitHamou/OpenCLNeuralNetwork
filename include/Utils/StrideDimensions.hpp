#include "Utils/Dimensions.hpp"
namespace Utils {
    struct StrideDimensions : public Dimensions {
        StrideDimensions(std::vector<size_t> p_dimensions) {
            if (p_dimensions.size() == 2) {
                for (const auto& dim : p_dimensions) {
                    if (dim <= 0) {
                        throw std::invalid_argument("Dimensions cannot be zero or negative.");
                    }
                }
                m_dimensions = p_dimensions;
            }
            else {
                throw std::invalid_argument("StrideDimensions requires 2-dimensional vector.");
            }
        }

        StrideDimensions(std::initializer_list<size_t> p_dimensions) {
            if (p_dimensions.size() == 2) {
                for (const auto& dim : p_dimensions) {
                    if (dim <= 0) {
                        throw std::invalid_argument("Dimensions cannot be zero or negative.");
                    }
                }
                m_dimensions = p_dimensions;
            }
            else {
                throw std::invalid_argument("StrideDimensions requires 2-dimensional vector.");
            }
        }

        StrideDimensions(size_t p_strideHeight, size_t p_strideWidth)
            : Dimensions({p_strideHeight, p_strideWidth}) {}

        StrideDimensions() : StrideDimensions(1, 1) {}

        size_t getHeight() const { return m_dimensions[0]; }
        size_t getWidth() const { return m_dimensions[1]; }
    };
}
