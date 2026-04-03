#pragma once

#include "Visualization/VisualizationTensor.hpp"
#include "Utils/LayerType.hpp"

#include <vector>
#include <string>

namespace Visualization
{
    struct LayerVisualization
    {
        size_t m_layerId;
        Utils::LayerType m_layerType;
        std::vector<std::string> m_metadata;
        std::vector<VisualizationTensor> m_visualizationTensors;

        std::string toString() const
        {
            std::string result = "Layer " + std::to_string(m_layerId) + ": " + Utils::layerTypeToString(m_layerType) + "\n";
            for (const std::string &meta : m_metadata)
            {
                result += "  " + meta + "\n";
            }
            for (const VisualizationTensor &tensor : m_visualizationTensors)
            {
                result += tensor.toString() + "\n";
            }
            return result;
        }
    };
}