#include "HelperFunctions.clh"

__kernel void computeLossGradient(
    __global const float* p_outputs,
    __global float* p_deltas,
    const unsigned int p_lossFunctionType,
    const unsigned int p_totalOutputElements,
    __global const float* p_targets
) {
    const unsigned int idx = get_global_id(1) * p_totalOutputElements + get_global_id(0);
    p_deltas[idx] = computeLossDerivative(p_outputs[idx], p_targets[idx], p_lossFunctionType);
}

__kernel void computePolicyGradient(
    __global const float* p_outputs,
    __global float* p_deltas,
    const unsigned int p_totalOutputElements,
    __global const int* p_actions,
    __global const float* p_rewards
) {
    int outputIdx = get_global_id(0);
    int batchIdx  = get_global_id(1);
    int index = batchIdx * p_totalOutputElements + outputIdx;
    float outputVal = p_outputs[index];
    int chosenAction = p_actions[batchIdx];
    p_deltas[index] = p_rewards[batchIdx] * ((outputIdx == chosenAction) * (1.0f - outputVal) + (outputIdx != chosenAction) * (-outputVal));
}