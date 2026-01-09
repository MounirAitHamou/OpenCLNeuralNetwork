#include "Layers/ActivationLayers/PreActivationLayers/LeakyReLU/LeakyReLULayer.hpp"
namespace Layers::Activation {
    void LeakyReLULayer::setupKernels() {
        cl_int err;

        m_forwardKernel = cl::Kernel(m_sharedResources->getProgram(), "leakyReLUForward", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create ReLU forward kernel");
        }
        m_forwardKernel.setArg(1, getOutputs());
        m_forwardKernel.setArg(2, getPreActivations());
        m_forwardKernel.setArg(3, getAlpha());

        m_backwardKernel = cl::Kernel(m_sharedResources->getProgram(), "leakyReLUBackward", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create ReLU backward kernel");
        }
        m_backwardKernel.setArg(1, getDeltas());
        m_backwardKernel.setArg(2, getPreActivations());
        m_backwardKernel.setArg(3, getAlpha());
    }
}