#include "Optimizers/SGD/SGDOptimizer.hpp"
namespace Optimizers
{
    void SGDOptimizer::setupKernels()
    {
        m_updateKernel = cl::Kernel(m_sharedResources->getProgram(), "sgdUpdateParameters");
        Utils::setKernelArgs(2, m_updateKernel, m_learningRate, m_weightDecayRate);
    }

    cl::Event SGDOptimizer::updateParameters(const cl::CommandQueue &p_concurrentQueue,
                                             const cl::Event p_lastEvent,
                                             const std::string &,
                                             cl::Buffer &p_parameters,
                                             cl::Buffer &p_gradients,
                                             size_t p_numElements)
    {
        Utils::setKernelArgs(m_updateKernel, p_parameters, p_gradients);
        cl::Event kernelEvent;
        std::vector<cl::Event> eventList = {p_lastEvent};
        p_concurrentQueue.enqueueNDRangeKernel(m_updateKernel, cl::NullRange,
                                               cl::NDRange(p_numElements), cl::NullRange, &eventList, &kernelEvent);

        return kernelEvent;
    }
}