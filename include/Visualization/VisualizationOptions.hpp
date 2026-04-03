#pragma once
#include <vector>
#include <string>

namespace Visualization
{
    enum class BufferType
    {
        Outputs,
        Deltas,
        PreActivations,
        Weights,
        Biases,
        WeightGradients,
        BiasGradients
    };

    inline std::string bufferTypeToString(Visualization::BufferType bufferType)
    {
        switch (bufferType)
        {
        case Visualization::BufferType::Outputs:
            return "Outputs";
        case Visualization::BufferType::Deltas:
            return "Deltas";
        case Visualization::BufferType::PreActivations:
            return "PreActivations";
        case Visualization::BufferType::Weights:
            return "Weights";
        case Visualization::BufferType::Biases:
            return "Biases";
        case Visualization::BufferType::WeightGradients:
            return "WeightGradients";
        case Visualization::BufferType::BiasGradients:
            return "BiasGradients";
        default:
            return "Unknown";
        }
    }

    struct VisualizationOptions
    {
        std::vector<BufferType> m_buffersToRead;
        int sampleIndex = -1;
    };
}