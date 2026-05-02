#include "LossFunctions/SoftmaxCrossEntropy/SoftmaxCrossEntropy.hpp"

namespace LossFunctions
{
    cl::Event SoftmaxCrossEntropy::computeLossGradient(const cl::CommandQueue &p_queue,
                                                       const cl::Buffer &p_predictions,
                                                       const cl::Buffer &p_targets,
                                                       cl::Buffer &p_outputGradients,
                                                       const size_t p_outputElements,
                                                       const size_t p_batchSize)
    {
        Utils::setKernelArgs(m_gradientKernel, p_predictions, p_targets, p_outputGradients, static_cast<cl_uint>(p_outputElements));
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

    void SoftmaxCrossEntropy::setupKernel()
    {
        cl_int err;
        m_gradientKernel = cl::Kernel(m_sharedResources->getProgram(), "softmaxCrossEntropyComputeGradients", &err);
        if (err != CL_SUCCESS)
        {
            CLNN_FATAL("Failed to create SoftmaxCrossEntropy gradient kernel");
        }
    }

    float SoftmaxCrossEntropy::computeLoss(
        const std::vector<float> &p_predictions,
        const std::vector<float> &p_targets,
        size_t p_outputElements,
        size_t p_batchSize)
    {
        float totalLoss = 0.0F;

        for (size_t batchIndex = 0; batchIndex < p_batchSize; ++batchIndex)
        {
            float sampleLoss = 0.0F;
            size_t base = batchIndex * p_outputElements;

            for (size_t sampleIndex = 0; sampleIndex < p_outputElements; ++sampleIndex)
            {
                float eps = kEpsilon;
                float pred = std::max(p_predictions[base + sampleIndex], eps);
                sampleLoss += -p_targets[base + sampleIndex] * std::log(pred);
            }

            totalLoss += sampleLoss;
        }

        return totalLoss / static_cast<float>(p_batchSize);
    }
}