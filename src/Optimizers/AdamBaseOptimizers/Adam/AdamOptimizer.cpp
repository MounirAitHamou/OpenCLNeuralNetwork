#include "Optimizers/AdamBaseOptimizers/Adam/AdamOptimizer.hpp"
namespace Optimizers {
    void AdamOptimizer::setupKernels() {
        m_updateKernel = cl::Kernel(m_sharedResources->getProgram(), "adamUpdateParameters");
        m_updateKernel.setArg(4, m_learningRate);
        m_updateKernel.setArg(5, m_beta1);
        m_updateKernel.setArg(6, m_beta2);
        m_updateKernel.setArg(7, m_epsilon);
        m_updateKernel.setArg(8, m_weightDecayRate);
    }
}
