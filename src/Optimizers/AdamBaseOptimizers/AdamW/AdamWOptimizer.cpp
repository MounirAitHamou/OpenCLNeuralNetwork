#include "Optimizers/AdamBaseOptimizers/AdamW/AdamWOptimizer.hpp"
namespace Optimizers
{
    void AdamWOptimizer::setupKernels()
    {
        m_updateKernel = cl::Kernel(m_sharedResources->getProgram(), "adamWUpdateParameters");
        Utils::setKernelArgs(4, m_updateKernel, m_learningRate, m_beta1, m_beta2, m_epsilon, m_weightDecayRate);
    }
}