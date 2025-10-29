#include "HelperFunctions.clh"

__kernel void computeLossGradient(
    __global const float* p_targets,
    __global const float* p_outputs,
    __global float* p_deltas,
    const unsigned int p_lossFunctionType,
    const unsigned int p_totalOutputElements
) {
    const unsigned int idx = get_global_id(1) * p_totalOutputElements + get_global_id(0);
    p_deltas[idx] = computeLossDerivative(p_outputs[idx], p_targets[idx], p_lossFunctionType);
}
