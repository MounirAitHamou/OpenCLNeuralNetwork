#pragma once

#include <CL/opencl.hpp>
#include <vector>
#include <Utils/Dimensions.hpp>

struct Batch {
public:
    Batch(cl::Buffer p_inputs, 
          cl::Buffer p_targets,
          std::vector<float> p_inputVec,
          std::vector<float> p_targetVec,
          size_t p_size,
          const Utils::Dimensions& p_inputDimensions,
          const Utils::Dimensions& p_targetDimensions)
        : m_inputs(std::move(p_inputs)),
          m_targets(std::move(p_targets)),
          m_inputsVec(std::move(p_inputVec)),
          m_targetsVec(std::move(p_targetVec)),
          m_size(p_size),
          m_inputDimensions(p_inputDimensions),
          m_targetDimensions(p_targetDimensions) {
            m_hasTargets = true;
          }

          

    const cl::Buffer& getInputs() const {
        return m_inputs;
    }

    const cl::Buffer& getTargets() const {
        return m_targets;
    }

    const std::vector<float>& getInputsVector() const {
        return m_inputsVec;
    }

    const std::vector<float>& getTargetsVector() const {
        return m_targetsVec;
    }

    size_t getSize() const {
        return m_size;
    }

    const Utils::Dimensions& getInputDimensions() const {
        return m_inputDimensions;
    }

    const Utils::Dimensions& getTargetDimensions() const {
        return m_targetDimensions;
    }

    bool hasTargets() const {
        return m_hasTargets;
    }


private:
    cl::Buffer m_inputs;
    cl::Buffer m_targets;
    std::vector<float> m_inputsVec;
    std::vector<float> m_targetsVec;
    size_t m_size;
    Utils::Dimensions m_inputDimensions;
    Utils::Dimensions m_targetDimensions;
    bool m_hasTargets;
};