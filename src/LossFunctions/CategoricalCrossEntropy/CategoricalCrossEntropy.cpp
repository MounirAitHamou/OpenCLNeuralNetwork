#include "LossFunctions/CategoricalCrossEntropy/CategoricalCrossEntropy.hpp"

namespace LossFunctions
{
    cl::Event CategoricalCrossEntropy::computeLossGradient(const cl::CommandQueue &p_queue,
                                                           const cl::Buffer &p_predictions,
                                                           const cl::Buffer &p_targets,
                                                           cl::Buffer &p_outputGradients,
                                                           const size_t p_outputElements,
                                                           const size_t p_batchSize)
    {
        Utils::setKernelArgs(m_gradientKernel, p_predictions, p_targets, p_outputGradients, (cl_uint)p_outputElements, (cl_uint)p_batchSize);
        cl::NDRange global(p_batchSize, p_outputElements);
        cl::Event kernelEvent;
        cl_int err = p_queue.enqueueNDRangeKernel(m_gradientKernel,
                                                  cl::NullRange,
                                                  global,
                                                  cl::NullRange,
                                                  nullptr,
                                                  &kernelEvent);
        if (err != CL_SUCCESS)
        {
            std::cout << "Error code: " << err << "\n";
            throw std::runtime_error("Failed to enqueue CategoricalCrossEntropy gradient kernel. Error code: " + std::to_string(err));
        }

        return kernelEvent;
    }

    void CategoricalCrossEntropy::setupKernel()
    {
        cl_int err;
        m_gradientKernel = cl::Kernel(m_sharedResources->getProgram(), "categoricalCrossEntropyComputeGradients", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create CategoricalCrossEntropy gradient kernel");
        }
    }

    float CategoricalCrossEntropy::computeLoss(
        const std::vector<float> &p_predictions,
        const std::vector<float> &p_targets,
        size_t p_outputElements,
        size_t p_batchSize)
    {
        float totalLoss = 0.0f;
        float eps = 1e-7f;
        for (size_t b = 0; b < p_batchSize; ++b)
        {
            float sampleLoss = 0.0f;
            size_t base = b * p_outputElements;

            for (size_t c = 0; c < p_outputElements; ++c)
            {
                if (p_targets[base + c] > 0.0f)
                {
                    float pred = std::clamp(p_predictions[base + c], eps, 1.0f);
                    sampleLoss = -std::log(pred);
                    break;
                }
            }

            totalLoss += sampleLoss;
        }

        return totalLoss / static_cast<float>(p_batchSize);
    }
}