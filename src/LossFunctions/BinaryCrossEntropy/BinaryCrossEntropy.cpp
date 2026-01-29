#include "LossFunctions/BinaryCrossEntropy/BinaryCrossEntropy.hpp"

namespace LossFunctions {
    cl::Event BinaryCrossEntropy::computeLossGradient(const cl::CommandQueue& p_queue,
                                                        const cl::Buffer& p_predictions, 
                                                        const cl::Buffer& p_targets, 
                                                        cl::Buffer& p_outputGradients, 
                                                        const size_t p_outputElements, 
                                                        const size_t p_batchSize) {
        m_gradientKernel.setArg(0, p_predictions);
        m_gradientKernel.setArg(1, p_targets);
        m_gradientKernel.setArg(2, p_outputGradients);
        m_gradientKernel.setArg(3, static_cast<cl_uint>(p_outputElements));
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

    void BinaryCrossEntropy::setupKernel() {
        cl_int err;
        m_gradientKernel = cl::Kernel(m_sharedResources->getProgram(), "binaryCrossEntropyComputeGradients", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create BinaryCrossEntropy gradient kernel");
        }
    }

    float BinaryCrossEntropy::computeLoss(const std::vector<float>& p_predictions, 
                                            const std::vector<float>& p_targets, 
                                            size_t p_outputElements, 
                                            size_t p_batchSize) {
        float totalLoss = 0.0f;
        size_t totalElements = p_outputElements * p_batchSize;
        for (size_t i = 0; i < totalElements; ++i) {
            float pred = std::min(std::max(p_predictions[i], 1e-15f), 1.0f - 1e-15f);
            totalLoss += - (p_targets[i] * std::log(pred) + (1.0f - p_targets[i]) * std::log(1.0f - pred));
        }
        return totalLoss / static_cast<float>(p_batchSize);
    }
}