#include "Layers/ActivationLayers/Sigmoid/SigmoidLayer.hpp"
namespace Layers::Activation
{
    void SigmoidLayer::setupKernels()
    {
        cl_int err;

        m_forwardKernel = cl::Kernel(m_sharedResources->getProgram(), "sigmoidForward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create Sigmoid forward kernel");
        }
        Utils::setKernelArgs(1, m_forwardKernel, getOutputs());

        m_backwardKernel = cl::Kernel(m_sharedResources->getProgram(), "sigmoidBackward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create Sigmoid backward kernel");
        }
        Utils::setKernelArgs(1, m_backwardKernel, getDeltas(), getOutputs());
    }
}