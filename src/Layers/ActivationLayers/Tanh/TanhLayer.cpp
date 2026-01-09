#include "Layers/ActivationLayers/Tanh/TanhLayer.hpp"
namespace Layers::Activation {
    void TanhLayer::setupKernels() {
        cl_int err;

        m_forwardKernel = cl::Kernel(m_sharedResources->getProgram(), "tanhForward", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create Tanh forward kernel");
        }
        m_forwardKernel.setArg(1, getOutputs());

        m_backwardKernel = cl::Kernel(m_sharedResources->getProgram(), "tanhBackward", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create Tanh backward kernel");
        }
        m_backwardKernel.setArg(1, getDeltas());
        m_backwardKernel.setArg(2, getOutputs());
    }
}