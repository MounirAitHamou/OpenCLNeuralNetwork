#include "Optimizers/AdamBaseOptimizers/Adam/AdamOptimizer.hpp"
namespace Optimizers
{
    void AdamOptimizer::setupKernels()
    {
        m_updateKernel = cl::Kernel(m_sharedResources->getProgram(), "adamUpdateParameters");
        Utils::setKernelArgs(4, m_updateKernel, m_learningRate, m_beta1, m_beta2, m_epsilon, m_weightDecayRate);
    }
}
