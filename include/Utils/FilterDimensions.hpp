#pragma once
#include "Utils/Dimensions.hpp"

namespace Utils {

    struct FilterDimensions : public Dimensions {
        FilterDimensions(std::vector<size_t> p_dimensions) {
            if (p_dimensions.size() == 4) {
                for (const auto& dim : p_dimensions) {
                    if (dim <= 0) {
                        throw std::invalid_argument("Dimensions cannot be zero or negative.");
                    }
                }
                m_dimensions = p_dimensions;
            }
            else {
                throw std::invalid_argument("FilterDimensions requires 4-dimensional vector.");
            }
        }

        FilterDimensions(std::initializer_list<size_t> p_dimensions) {
            if (p_dimensions.size() == 4) {
                for (const auto& dim : p_dimensions) {
                    if (dim <= 0) {
                        throw std::invalid_argument("Dimensions cannot be zero or negative.");
                    }
                }
                m_dimensions = p_dimensions;
            }
            else {
                throw std::invalid_argument("FilterDimensions requires 4-dimensional vector.");
            }
        }

        FilterDimensions(size_t p_filterHeight, size_t p_filterWidth, size_t p_inputChannels, size_t p_outputChannels)
            : Dimensions({p_filterHeight, p_filterWidth, p_inputChannels, p_outputChannels}) {}

        FilterDimensions() : FilterDimensions(1, 1, 1, 1) {}

        size_t getHeight() const { return m_dimensions[0]; }
        size_t getWidth() const { return m_dimensions[1]; }
        size_t getInputChannels() const { return m_dimensions[2]; }
        size_t getOutputChannels() const { return m_dimensions[3]; }
    };
}