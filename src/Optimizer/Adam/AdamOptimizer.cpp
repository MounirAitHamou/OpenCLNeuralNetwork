#include "Optimizer/Adam/AdamOptimizer.hpp"

AdamOptimizer::AdamOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                             float p_learningRate,
                             float p_weightDecayRate,
                             float p_beta1,
                             float p_beta2,
                             float p_epsilon)
        : Optimizer(p_sharedResources, p_learningRate, p_weightDecayRate),
          m_beta1(p_beta1), m_beta2(p_beta2), m_epsilon(p_epsilon), m_t(1) {setupKernels();}

AdamOptimizer::AdamOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                             const H5::Group& p_optimizerGroup)
    :Optimizer(p_sharedResources, p_optimizerGroup) {
    p_optimizerGroup.openAttribute("beta1").read(H5::PredType::NATIVE_FLOAT, &m_beta1);
    p_optimizerGroup.openAttribute("beta2").read(H5::PredType::NATIVE_FLOAT, &m_beta2);
    p_optimizerGroup.openAttribute("epsilon").read(H5::PredType::NATIVE_FLOAT, &m_epsilon);
    p_optimizerGroup.openAttribute("t").read(H5::PredType::NATIVE_UINT, &m_t);
    
    H5::Group momentBuffersGroup = p_optimizerGroup.openGroup("momentBuffers");
    
    hsize_t numLayers = momentBuffersGroup.getNumObjs();
    for (hsize_t i = 0; i < numLayers; ++i) {
        std::string layerName = momentBuffersGroup.getObjnameByIdx(i);
        H5::Group layerGroup = momentBuffersGroup.openGroup(layerName);
        size_t layerId;
        layerGroup.openAttribute("layerId").read(H5::PredType::NATIVE_HSIZE, &layerId);

        size_t weightsSize, biasesSize;
        layerGroup.openAttribute("weightsSize").read(H5::PredType::NATIVE_HSIZE, &weightsSize);
        layerGroup.openAttribute("biasesSize").read(H5::PredType::NATIVE_HSIZE, &biasesSize);
        if (weightsSize > 0 && layerGroup.nameExists("weightsFirstMomentBuffer") == true &&
                layerGroup.nameExists("weightsSecondMomentBuffer") == true) {
            std::string weightKey = std::to_string(layerId) + "Weights";
            cl::Buffer mBuffer = Utils::loadBuffer(m_sharedResources->getContext(), layerGroup, "weightsFirstMomentBuffer", weightsSize);
            cl::Buffer vBuffer = Utils::loadBuffer(m_sharedResources->getContext(), layerGroup, "weightsSecondMomentBuffer", weightsSize);
            m_momentBuffers[weightKey] = {mBuffer, vBuffer};
        }

        if (biasesSize > 0 && layerGroup.nameExists("biasesFirstMomentBuffer") == true &&
                layerGroup.nameExists("biasesSecondMomentBuffer") == true) {
            std::string biasKey = std::to_string(layerId) + "Biases";
            cl::Buffer mBuffer = Utils::loadBuffer(m_sharedResources->getContext(), layerGroup, "biasesFirstMomentBuffer", biasesSize);
            cl::Buffer vBuffer = Utils::loadBuffer(m_sharedResources->getContext(), layerGroup, "biasesSecondMomentBuffer", biasesSize);
            m_momentBuffers[biasKey] = {mBuffer, vBuffer};
        }
    }
    setupKernels();
}

bool AdamOptimizer::equals(const cl::CommandQueue& p_queue, const Optimizer& p_other, std::map<size_t, std::pair<size_t, size_t>>& p_momentsSizes) const {
    if (p_other.getType() != Utils::OptimizerType::Adam) {
        return false;
    }
    const AdamOptimizer& otherAdam = static_cast<const AdamOptimizer&>(p_other);

    if (m_learningRate != otherAdam.m_learningRate ||
        m_weightDecayRate != otherAdam.m_weightDecayRate ||
        m_beta1 != otherAdam.m_beta1 ||
        m_beta2 != otherAdam.m_beta2 ||
        m_epsilon != otherAdam.m_epsilon) {
        return false;
    }

    if (m_momentBuffers.size() != otherAdam.m_momentBuffers.size()) {
        return false;
    }
    
    for (const auto& [layerId, sizes] : p_momentsSizes) {
        std::string layerIdStr = std::to_string(layerId);
        
        if (!compareBuffers(p_queue, layerIdStr + "Weights", sizes.first, m_momentBuffers, otherAdam.m_momentBuffers)) {
            return false;
        }
        
        if (!compareBuffers(p_queue, layerIdStr + "Biases", sizes.second, m_momentBuffers, otherAdam.m_momentBuffers)) {
            return false;
        }
    }
    
    return true;
}


void AdamOptimizer::setupKernels() {
    m_updateKernel = cl::Kernel(m_sharedResources->getProgram(), "adamUpdateParameters");
    m_updateKernel.setArg(4, m_learningRate);
    m_updateKernel.setArg(5, m_beta1);
    m_updateKernel.setArg(6, m_beta2);
    m_updateKernel.setArg(7, m_epsilon);
    m_updateKernel.setArg(8, m_weightDecayRate);
}

cl::Event AdamOptimizer::updateParameters(const cl::CommandQueue& p_concurrentQueue, 
                               cl::Event& p_lastEvent, 
                               const std::string& p_parametersId, 
                               cl::Buffer& p_parameters, 
                               cl::Buffer& p_gradients, 
                               size_t p_numElements) {
    auto foundBuffers = m_momentBuffers.find(p_parametersId);
    cl::Buffer mBuffer;
    cl::Buffer vBuffer;

    if (foundBuffers == m_momentBuffers.end()) {
        std::vector<float> zeroData(p_numElements, 0.0f);

        mBuffer = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           p_numElements * sizeof(float), zeroData.data());

        vBuffer = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           p_numElements * sizeof(float), zeroData.data());

        m_momentBuffers[p_parametersId] = {mBuffer, vBuffer};
    } else {
        mBuffer = foundBuffers->second.first;
        vBuffer = foundBuffers->second.second;
    }

    m_updateKernel.setArg(0, p_parameters);
    m_updateKernel.setArg(1, p_gradients);
    m_updateKernel.setArg(2, mBuffer);
    m_updateKernel.setArg(3, vBuffer);
    m_updateKernel.setArg(9, pow(m_beta1, (float)m_t));
    m_updateKernel.setArg(10, pow(m_beta2, (float)m_t));

    cl::Event kernelEvent;
    std::vector<cl::Event> eventList = {p_lastEvent};

    p_concurrentQueue.enqueueNDRangeKernel(m_updateKernel, cl::NullRange,
                               cl::NDRange(p_numElements), cl::NullRange, &eventList, &kernelEvent);

    return kernelEvent;
}

void AdamOptimizer::step() {
    m_t++;
}

void AdamOptimizer::saveOptimizer(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup,
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
    p_optimizerGroup.createAttribute("beta1", H5::PredType::NATIVE_FLOAT, scalarDataspace)
        .write(H5::PredType::NATIVE_FLOAT, &m_beta1);
    p_optimizerGroup.createAttribute("beta2", H5::PredType::NATIVE_FLOAT, scalarDataspace)
        .write(H5::PredType::NATIVE_FLOAT, &m_beta2);
    p_optimizerGroup.createAttribute("epsilon", H5::PredType::NATIVE_FLOAT, scalarDataspace)
        .write(H5::PredType::NATIVE_FLOAT, &m_epsilon);
    p_optimizerGroup.createAttribute("t", H5::PredType::NATIVE_UINT, scalarDataspace)
        .write(H5::PredType::NATIVE_UINT, &m_t);

    H5::Group momentBuffersGroup = p_optimizerGroup.createGroup("momentBuffers");
    for (const auto& [layerId, sizes] : p_momentsSizes) {
        size_t weightsSize = sizes.first;
        size_t biasesSize = sizes.second;
        std::string layerIdStr = std::to_string(layerId);
        H5::Group layerGroup = momentBuffersGroup.createGroup(layerIdStr);
        layerGroup.createAttribute("layerId", H5::PredType::NATIVE_HSIZE, scalarDataspace)
            .write(H5::PredType::NATIVE_HSIZE, &layerId);
        layerGroup.createAttribute("weightsSize", H5::PredType::NATIVE_HSIZE, scalarDataspace)
            .write(H5::PredType::NATIVE_HSIZE, &weightsSize);
        layerGroup.createAttribute("biasesSize", H5::PredType::NATIVE_HSIZE, scalarDataspace)
            .write(H5::PredType::NATIVE_HSIZE, &biasesSize);
        if (weightsSize > 0) {
            std::string weightKey = layerIdStr + "Weights";
            auto buffersIt = m_momentBuffers.find(weightKey);
            if (buffersIt != m_momentBuffers.end()) {
                Utils::saveBuffer(p_queue, buffersIt->second.first, layerGroup, "weightsFirstMomentBuffer", weightsSize);
                Utils::saveBuffer(p_queue, buffersIt->second.second, layerGroup, "weightsSecondMomentBuffer", weightsSize);
            }

        }

        if (biasesSize > 0) {
            std::string biasKey = layerIdStr + "Biases";
            auto buffersIt = m_momentBuffers.find(biasKey);
            if (buffersIt != m_momentBuffers.end()) {
                Utils::saveBuffer(p_queue, buffersIt->second.first, layerGroup, "biasesFirstMomentBuffer", biasesSize);
                Utils::saveBuffer(p_queue, buffersIt->second.second, layerGroup, "biasesSecondMomentBuffer", biasesSize);
            }
        }
    }
}