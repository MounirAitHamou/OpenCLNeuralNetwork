#pragma once

#include <stdexcept>
#include <vector>
#include <numeric>
#include <string>
#include <sstream>

namespace Utils {

    struct Dimensions {
    protected:
        std::vector<size_t> m_dimensions;

    public:
        Dimensions() = default;

        Dimensions(std::initializer_list<size_t> p_dimensions) {
            for (const auto& dim : p_dimensions) {
                if (dim <= 0) {
                    throw std::invalid_argument("Dimensions cannot be zero or negative.");
                }
            }
            m_dimensions = p_dimensions;
        }

        Dimensions(const std::vector<size_t>& p_dimensions) {
            for (const auto& dim : p_dimensions) {
                if (dim <= 0) {
                    throw std::invalid_argument("Dimensions cannot be zero or negative.");
                }
            }
            m_dimensions = p_dimensions;
        }

        std::vector<size_t> getDimensions() const { return m_dimensions; }

        size_t getTotalElements() const {
            if (m_dimensions.empty()) {
                return 0;
            }
            return std::accumulate(m_dimensions.begin(), m_dimensions.end(), (size_t)1, std::multiplies<size_t>());
        }

        bool operator==(const Dimensions& p_other) const {
            return m_dimensions == p_other.m_dimensions;
        }

        bool operator!=(const Dimensions& p_other) const {
            return !(*this == p_other);
        }

        std::string toString() const {
            if (m_dimensions.empty()) {
                return "";
            }

            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < m_dimensions.size(); ++i) {
                oss << m_dimensions[i];
                if (i < m_dimensions.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "]";
            return oss.str();
        }

        static Dimensions validateDenseDimensions(const Dimensions& p_outputDimensions) {
            if (p_outputDimensions.getDimensions().size() != 1) {
                throw std::invalid_argument("Dense layer requires single-dimensional output dimensions.");
            }
            return p_outputDimensions;
        }
    };
}