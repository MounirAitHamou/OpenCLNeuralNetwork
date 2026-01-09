#include "Optimizers/SGD/SGDOptimizer.hpp"
namespace Optimizers {
    void SGDOptimizer::setupKernels() {
        m_updateKernel = cl::Kernel(m_sharedResources->getProgram(), "sgdUpdateParameters");
        m_updateKernel.setArg(2, m_learningRate);
        m_updateKernel.setArg(3, m_weightDecayRate);
    }

    cl::Event SGDOptimizer::updateParameters(const cl::CommandQueue& p_concurrentQueue, 
                                            const cl::Event& p_lastEvent, 
                                            const std::string& p_parametersId, 
                                            cl::Buffer& p_parameters, 
                                            cl::Buffer& p_gradients, 
                                            size_t p_numElements) {
        m_updateKernel.setArg(0, p_parameters);
        m_updateKernel.setArg(1, p_gradients);
        
        cl::Event kernelEvent;
        std::vector<cl::Event> eventList = {p_lastEvent};
        p_concurrentQueue.enqueueNDRangeKernel(m_updateKernel, cl::NullRange,
                                cl::NDRange(p_numElements), cl::NullRange, &eventList, &kernelEvent);
        
        return kernelEvent;
    }
}