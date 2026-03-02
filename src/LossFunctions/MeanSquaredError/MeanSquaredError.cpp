#include "LossFunctions/MeanSquaredError/MeanSquaredError.hpp"

namespace LossFunctions
{
    cl::Event MeanSquaredError::computeLossGradient(const cl::CommandQueue &p_queue,
                                                    const cl::Buffer &p_predictions,
                                                    const cl::Buffer &p_targets,
                                                    cl::Buffer &p_outputGradients,
                                                    const size_t p_outputElements,
                                                    const size_t p_batchSize)
    {
        Utils::setKernelArgs(m_gradientKernel, p_predictions, p_targets, p_outputGradients, (cl_uint)p_outputElements);
        cl::NDRange global(p_batchSize, p_outputElements);
        cl::Event kernelEvent;
        p_queue.enqueueNDRangeKernel(m_gradientKernel,
                                     cl::NullRange,
                                     global,
                                     cl::NullRange,
                                     nullptr,
                                     &kernelEvent);
        return kernelEvent;
    }

    void MeanSquaredError::setupKernel()
    {
        cl_int err;
        m_gradientKernel = cl::Kernel(m_sharedResources->getProgram(), "meanSquaredErrorComputeGradients", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create MeanSquaredError gradient kernel");
        }
    }

    float MeanSquaredError::computeLoss(const std::vector<float> &p_predictions,
                                        const std::vector<float> &p_targets,
                                        size_t p_outputElements,
                                        size_t p_batchSize)
    {
        float totalLoss = 0.0f;
        size_t totalElements = p_outputElements * p_batchSize;
        for (size_t i = 0; i < totalElements; ++i)
        {
            float diff = p_predictions[i] - p_targets[i];
            totalLoss += diff * diff;
        }
        return totalLoss / static_cast<float>(totalElements);
    }
}