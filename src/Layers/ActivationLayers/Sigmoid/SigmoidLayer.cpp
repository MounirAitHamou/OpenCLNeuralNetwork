#include "Layers/ActivationLayers/Sigmoid/SigmoidLayer.hpp"
namespace Layers::Activation {
    void SigmoidLayer::setupKernels() {
        cl_int err;

        m_forwardKernel = cl::Kernel(m_sharedResources->getProgram(), "sigmoidForward", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create Sigmoid forward kernel");
        }
        m_forwardKernel.setArg(1, getOutputs());

        m_backwardKernel = cl::Kernel(m_sharedResources->getProgram(), "sigmoidBackward", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create Sigmoid backward kernel");
        }
        m_backwardKernel.setArg(1, getDeltas());
        m_backwardKernel.setArg(2, getOutputs());
    }
}