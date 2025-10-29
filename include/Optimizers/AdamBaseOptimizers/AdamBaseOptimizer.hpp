#pragma once

#include "Optimizers/Optimizer.hpp"

class AdamBaseOptimizer : public Optimizer {
public:
    AdamBaseOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                  float p_learningRate,
                  float p_weightDecayRate,
                  float p_beta1 = 0.9f,
                  float p_beta2 = 0.999f,
                  float p_epsilon = 1e-8f)
        : Optimizer(p_sharedResources, p_learningRate, p_weightDecayRate),
          m_beta1(p_beta1),
          m_beta2(p_beta2),
          m_epsilon(p_epsilon),
          m_t(1) {}

    AdamBaseOptimizer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                      const H5::Group& p_optimizerGroup)
        : Optimizer(p_sharedResources, p_optimizerGroup) {
          m_beta1 = Utils::readValueFromHDF5<float>(p_optimizerGroup, "beta1");
          m_beta2 = Utils::readValueFromHDF5<float>(p_optimizerGroup, "beta2");
          m_epsilon = Utils::readValueFromHDF5<float>(p_optimizerGroup, "epsilon");
          m_t = Utils::readValueFromHDF5<unsigned int>(p_optimizerGroup, "t");
          loadMomentBuffers(p_optimizerGroup);
    }
    
    ~AdamBaseOptimizer() = default;

    cl::Event updateParameters(const cl::CommandQueue& p_concurrentQueue, 
                               const cl::Event& p_lastEvent, 
                               const std::string& p_parametersId, 
                               cl::Buffer& p_parameters, 
                               cl::Buffer& p_gradients, 
                               size_t p_numElements) final override {
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

    void save(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup, const std::map<size_t, std::pair<size_t, size_t>>& p_parameterSizes) const override { saveAdamBaseOptimizer(p_queue, p_optimizerGroup, p_parameterSizes); }
    bool equals(const cl::CommandQueue& p_queue, const Optimizer& p_other, std::map<size_t, std::pair<size_t, size_t>>& p_parameterSizes) const override { return adamBaseOptimizerEquals(p_queue, p_other, p_parameterSizes); }
    void print() const override { printAdamBaseOptimizer(); }

    void step() final override { m_t++; }

protected:
    float m_beta1;
    float m_beta2;
    float m_epsilon;
    unsigned int m_t;

    std::map<std::string, std::pair<cl::Buffer, cl::Buffer>> m_momentBuffers;

    void saveAdamBaseOptimizer(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup,
        const std::map<size_t, std::pair<size_t, size_t>>& p_momentSizes) const {
            saveOptimizer(p_queue, p_optimizerGroup);
            Utils::writeValueToHDF5<float>(p_optimizerGroup, "beta1", m_beta1);
            Utils::writeValueToHDF5<float>(p_optimizerGroup, "beta2", m_beta2);
            Utils::writeValueToHDF5<float>(p_optimizerGroup, "epsilon", m_epsilon);
            Utils::writeValueToHDF5<unsigned int>(p_optimizerGroup, "t", m_t);
            saveMomentBuffers(p_queue, p_optimizerGroup, p_momentSizes);
        }
    

    bool adamBaseOptimizerEquals(const cl::CommandQueue& p_queue, const Optimizer& p_other, std::map<size_t, std::pair<size_t, size_t>>& p_momentSizes) const {
        if (!optimizerEquals(p_queue, p_other)) return false;
        const AdamBaseOptimizer& otherAdamBase = static_cast<const AdamBaseOptimizer&>(p_other);

        return m_beta1 == otherAdamBase.m_beta1 &&
            m_beta2 == otherAdamBase.m_beta2 &&
            m_epsilon == otherAdamBase.m_epsilon && 
            m_momentBuffers.size() == otherAdamBase.m_momentBuffers.size() &&
            momentBuffersEqual(p_queue, otherAdamBase, p_momentSizes);
    }

    void printAdamBaseOptimizer() const {
        printOptimizer();
        std::cout << "Beta1: " << m_beta1 << "\n"
                  << "Beta2: " << m_beta2 << "\n"
                  << "Epsilon: " << m_epsilon << "\n";
    }

    void saveMomentBuffers(const cl::CommandQueue& p_queue, H5::Group& p_optimizerGroup, 
        const std::map<size_t, std::pair<size_t, size_t>>& p_momentSizes) const {
        H5::Group momentBuffersGroup = p_optimizerGroup.createGroup("momentBuffers");
        for (const auto& [layerId, sizes] : p_momentSizes) {
            size_t weightsSize = sizes.first;
            size_t biasesSize = sizes.second;
            std::string layerIdStr = std::to_string(layerId);
            H5::Group layerGroup = momentBuffersGroup.createGroup(layerIdStr);
            Utils::writeValueToHDF5<size_t>(layerGroup, "layerId", layerId);
            Utils::writeValueToHDF5<size_t>(layerGroup, "weightsSize", weightsSize);
            Utils::writeValueToHDF5<size_t>(layerGroup, "biasesSize", biasesSize);
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


    void loadMomentBuffers(const H5::Group& p_optimizerGroup) {
        H5::Group momentBuffersGroup = p_optimizerGroup.openGroup("momentBuffers");
        hsize_t numLayers = momentBuffersGroup.getNumObjs();
        for (hsize_t i = 0; i < numLayers; ++i) {
            std::string layerName = momentBuffersGroup.getObjnameByIdx(i);
            H5::Group layerGroup = momentBuffersGroup.openGroup(layerName);
            size_t layerId;
            layerId = Utils::readValueFromHDF5<size_t>(layerGroup, "layerId");
            std::string layerIdStr = std::to_string(layerId);
            size_t weightsSize, biasesSize;
            weightsSize = Utils::readValueFromHDF5<size_t>(layerGroup, "weightsSize");
            biasesSize = Utils::readValueFromHDF5<size_t>(layerGroup, "biasesSize");
            if (weightsSize > 0 && layerGroup.nameExists("weightsFirstMomentBuffer") &&
                    layerGroup.nameExists("weightsSecondMomentBuffer")) {
                std::string weightKey = layerIdStr + "Weights";
                cl::Buffer mBuffer = Utils::loadBuffer(m_sharedResources->getContext(), layerGroup, "weightsFirstMomentBuffer", weightsSize);
                cl::Buffer vBuffer = Utils::loadBuffer(m_sharedResources->getContext(), layerGroup, "weightsSecondMomentBuffer", weightsSize);
                m_momentBuffers[weightKey] = {mBuffer, vBuffer};
            }

            if (biasesSize > 0 && layerGroup.nameExists("biasesFirstMomentBuffer") &&
                    layerGroup.nameExists("biasesSecondMomentBuffer")) {
                std::string biasKey = layerIdStr + "Biases";
                cl::Buffer mBuffer = Utils::loadBuffer(m_sharedResources->getContext(), layerGroup, "biasesFirstMomentBuffer", biasesSize);
                cl::Buffer vBuffer = Utils::loadBuffer(m_sharedResources->getContext(), layerGroup, "biasesSecondMomentBuffer", biasesSize);
                m_momentBuffers[biasKey] = {mBuffer, vBuffer};
            }
        }
    }

    bool momentBuffersEqual(const cl::CommandQueue& p_queue, const AdamBaseOptimizer& p_otherAdamBase, const std::map<size_t, std::pair<size_t, size_t>>& p_momentSizes) const {
        for (const auto& [layerId, sizes] : p_momentSizes) {
            std::string layerIdStr = std::to_string(layerId);
            if (!compareMomentBuffers(p_queue, layerIdStr + "Weights", sizes.first, m_momentBuffers, p_otherAdamBase.m_momentBuffers) || 
                !compareMomentBuffers(p_queue, layerIdStr + "Biases", sizes.second, m_momentBuffers, p_otherAdamBase.m_momentBuffers)) return false;
        }
        return true;
    }

    bool compareMomentBuffers(const cl::CommandQueue& p_queue, const std::string& p_key, size_t p_size,
                        const std::map<std::string, std::pair<cl::Buffer, cl::Buffer>>& p_buffers1, 
                        const std::map<std::string, std::pair<cl::Buffer, cl::Buffer>>& p_buffers2) const {
        const float epsilon = 1e-6f;
        if (p_size == 0) {
            return true;
        }
        
        auto it1 = p_buffers1.find(p_key);
        auto it2 = p_buffers2.find(p_key);

        if ((it1 == p_buffers1.end() && it2 != p_buffers2.end()) || (it1 != p_buffers1.end() && it2 == p_buffers2.end())) { return false; }

        std::vector<float> data1 = Utils::readCLBuffer(p_queue, it1->second.first, p_size);
        std::vector<float> data2 = Utils::readCLBuffer(p_queue, it2->second.first, p_size);
        
        for (size_t i = 0; i < p_size; ++i) {
            if (fabs(data1[i] - data2[i]) > epsilon) {
                return false;
            }
        }

        std::vector<float> data1_v = Utils::readCLBuffer(p_queue, it1->second.second, p_size);
        std::vector<float> data2_v = Utils::readCLBuffer(p_queue, it2->second.second, p_size);
        for (size_t i = 0; i < p_size; ++i) {
            if (fabs(data1_v[i] - data2_v[i]) > epsilon) {
                return false;
            }
        }
        return true;
    }
};