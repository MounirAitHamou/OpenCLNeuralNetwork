#include "Layers/ActivationLayers/PreActivationLayers/ReLU/ReLULayer.hpp"
namespace Layers::Activation
{
    void ReLULayer::setupKernels()
    {
        cl_int err;

        m_forwardKernel = cl::Kernel(m_sharedResources->getProgram(), "reLUForward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create ReLU forward kernel");
        }
        Utils::setKernelArgs(1, m_forwardKernel, getOutputs(), getPreActivations());

        m_backwardKernel = cl::Kernel(m_sharedResources->getProgram(), "reLUBackward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create ReLU backward kernel");
        }
        Utils::setKernelArgs(1, m_backwardKernel, getDeltas(), getPreActivations());
    }
}