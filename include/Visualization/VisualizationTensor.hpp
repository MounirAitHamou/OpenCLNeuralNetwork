#pragma once

#include "Utils/Dimensions.hpp"
#include <string>

namespace Visualization
{
    struct VisualizationTensor
    {
        VisualizationTensor() = default;

        VisualizationTensor(std::string p_name, Utils::Dimensions p_dimensions, std::vector<float> p_data)
            : m_name(std::move(p_name)),
              m_dimensions(std::move(p_dimensions)),
              m_data(std::move(p_data))
        {
        }

        static VisualizationTensor createVisualizationTensor(
            const cl::CommandQueue &p_queue,
            const cl::Buffer &p_buffer,
            size_t p_offset,
            Utils::Dimensions p_dimensions,
            std::string p_name)
        {
            size_t totalElements = p_dimensions.getTotalElements();
            return VisualizationTensor(std::move(p_name), std::move(p_dimensions), Utils::readCLBuffer(p_queue, p_buffer, p_offset, totalElements));
        }

        std::string toString() const
        {
            std::string result = "Tensor: " + m_name + ", Dimensions: " + m_dimensions.toString() + ", Data: [";
            for (size_t i = 0; i < m_data.size(); ++i)
            {
                result += std::to_string(m_data[i]);
                if (i < m_data.size() - 1)
                {
                    result += ", ";
                }
            }
            result += "]";
            return result;
        }

        std::string m_name;
        Utils::Dimensions m_dimensions;
        std::vector<float> m_data;
    };
}