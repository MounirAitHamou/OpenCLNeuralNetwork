#include "Layers/ActivationLayers/Tanh/TanhLayer.hpp"
namespace Layers::Activation
{
    void TanhLayer::setupKernels()
    {
        cl_int err;

        m_forwardKernel = cl::Kernel(m_sharedResources->getProgram(), "tanhForward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create Tanh forward kernel");
        }
        Utils::setKernelArgs(1, m_forwardKernel, getOutputs());
        m_backwardKernel = cl::Kernel(m_sharedResources->getProgram(), "tanhBackward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create Tanh backward kernel");
        }
        Utils::setKernelArgs(1, m_backwardKernel, getDeltas(), getOutputs());
    }
}