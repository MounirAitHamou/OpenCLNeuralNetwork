#include "Layers/ActivationLayers/PreActivationLayers/LeakyReLU/LeakyReLULayer.hpp"
namespace Layers::Activation
{
    void LeakyReLULayer::setupKernels()
    {
        cl_int err;

        m_forwardKernel = cl::Kernel(m_sharedResources->getProgram(), "leakyReLUForward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create ReLU forward kernel");
        }
        Utils::setKernelArgs(1, m_forwardKernel, getOutputs(), getPreActivations(), getAlpha());

        m_backwardKernel = cl::Kernel(m_sharedResources->getProgram(), "leakyReLUBackward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create ReLU backward kernel");
        }
        Utils::setKernelArgs(1, m_backwardKernel, getDeltas(), getPreActivations(), getAlpha());
    }
}