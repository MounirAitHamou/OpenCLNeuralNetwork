#include "Optimizer/SGD/SGDOptimizer.hpp"

SGDOptimizer::SGDOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                           float p_learningRate,
                           float p_weightDecayRate)
    : Optimizer(p_sharedResources, p_learningRate, p_weightDecayRate) {
    setupKernels();
}

SGDOptimizer::SGDOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                           const H5::Group& p_optimizerGroup)
    :Optimizer(p_sharedResources, p_optimizerGroup) {
    setupKernels();
}

bool SGDOptimizer::equals(const cl::CommandQueue& p_queue, const Optimizer& p_other, std::map<size_t, std::pair<size_t, size_t>>& p_momentsSizes) const {
    if (p_other.getType() != Utils::OptimizerType::SGD) {
        return false;
    }
    const SGDOptimizer& otherSGD = static_cast<const SGDOptimizer&>(p_other);

    if (m_learningRate != otherSGD.m_learningRate ||
        m_weightDecayRate != otherSGD.m_weightDecayRate) {
        return false;
    }

    return true;
}

void SGDOptimizer::setupKernels() {
    m_updateKernel = cl::Kernel(m_sharedResources->getProgram(), "sgdUpdateParameters");
    m_updateKernel.setArg(2, m_learningRate);
    m_updateKernel.setArg(3, m_weightDecayRate);
}

cl::Event SGDOptimizer::updateParameters(const cl::CommandQueue& p_concurrentQueue, 
                                         cl::Event& p_lastEvent, 
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

void SGDOptimizer::saveOptimizer(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup,
        const std::map<size_t, std::pair<size_t, size_t>>& p_momentsSizes) const {
    H5::DataSpace scalarDataspace(H5S_SCALAR);
    unsigned int optimizerType = static_cast<unsigned int>(getType());
    p_optimizerGroup.createAttribute(
                "optimizerType", H5::PredType::NATIVE_UINT, scalarDataspace
            ).write(H5::PredType::NATIVE_UINT, &optimizerType);
    p_optimizerGroup.createAttribute("learningRate", H5::PredType::NATIVE_FLOAT, scalarDataspace)
        .write(H5::PredType::NATIVE_FLOAT, &m_learningRate);
    p_optimizerGroup.createAttribute("weightDecayRate", H5::PredType::NATIVE_FLOAT, scalarDataspace)
        .write(H5::PredType::NATIVE_FLOAT, &m_weightDecayRate);
}