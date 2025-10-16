#pragma once

#include <CL/opencl.hpp>

struct PolicyBatch {
public:
    PolicyBatch(cl::Buffer p_inputs,
                cl::Buffer p_actions,
                cl::Buffer p_rewards,
                size_t p_size)
        : m_inputs(std::move(p_inputs)),
          m_actions(std::move(p_actions)),
          m_rewards(std::move(p_rewards)),
          m_size(p_size) {}
    
    const cl::Buffer& getInputs() const {
        return m_inputs;
    }

    const cl::Buffer& getActions() const {
        return m_actions;
    }

    const cl::Buffer& getRewards() const {
        return m_rewards;
    }

    size_t getSize() const {
        return m_size;
    }

private:
    cl::Buffer m_inputs;
    cl::Buffer m_actions;
    cl::Buffer m_rewards;
    size_t m_size;
};