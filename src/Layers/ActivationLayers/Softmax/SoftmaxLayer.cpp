#include "Layers/ActivationLayers/Softmax/SoftmaxLayer.hpp"
namespace Layers::Activation
{
    void SoftmaxLayer::setupKernels()
    {
        cl_int err;

        m_forwardKernel = cl::Kernel(m_sharedResources->getProgram(), "softmaxForward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create Softmax forward kernel");
        }
        Utils::setKernelArgs(1, m_forwardKernel, getOutputs(), (cl_uint)getTotalOutputElements());

        m_backwardKernel = cl::Kernel(m_sharedResources->getProgram(), "softmaxBackward", &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Failed to create Softmax backward kernel");
        }
        Utils::setKernelArgs(1, m_backwardKernel, getDeltas(), getOutputs(), (cl_uint)getTotalOutputElements());
    }
}