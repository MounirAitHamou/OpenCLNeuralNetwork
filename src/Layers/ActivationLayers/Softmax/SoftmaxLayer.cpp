#include "Layers/ActivationLayers/Softmax/SoftmaxLayer.hpp"
namespace Layers::Activation {
    void SoftmaxLayer::setupKernels() {
        cl_int err;

        m_forwardKernel = cl::Kernel(m_sharedResources->getProgram(), "softmaxForward", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create Softmax forward kernel");
        }
        m_forwardKernel.setArg(1, getOutputs());
        m_forwardKernel.setArg(2, (cl_uint)getTotalOutputElements());

        m_backwardKernel = cl::Kernel(m_sharedResources->getProgram(), "softmaxBackward", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create Softmax backward kernel");
        }
        m_backwardKernel.setArg(1, getDeltas());
        m_backwardKernel.setArg(2, getOutputs());
        m_backwardKernel.setArg(3, (cl_uint)getTotalOutputElements());
    }
}