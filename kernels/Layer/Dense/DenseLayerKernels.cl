#include "HelperFunctions.clh"

__kernel void denseBiasActivation(
    __global float* p_preActivations,
    __global const float* p_biases,
    __global float* p_outputs,
    const unsigned int p_outputSize,
    const unsigned int p_activationType) {

    unsigned int outNeuronIdx = get_global_id(0);
    unsigned int idx = get_global_id(1) * p_outputSize + outNeuronIdx;
    p_preActivations[idx] += p_biases[outNeuronIdx];
    p_outputs[idx] = applyActivation(p_preActivations[idx], p_activationType);
}

__kernel void denseBackpropActivation(
    __global float* p_deltas,
    __global const float* p_preActivations,
    const unsigned int p_outputSize,
    const unsigned int p_activationType
){
    unsigned int idx = get_global_id(1) * p_outputSize + get_global_id(0);
    p_deltas[idx] *= applyActivationDerivative(p_preActivations[idx], p_activationType);
}

__kernel void denseAverageWeightsGradients(
    __global float* p_weightsGradients,
    const unsigned int p_inputSize,
    const unsigned int p_batchSize) {
        p_weightsGradients[get_global_id(0) * p_inputSize + get_global_id(1)] /= p_batchSize;
}

__kernel void denseAverageBiasesGradients(
    __global float* p_biasesGradients,
    const unsigned int p_batchSize) {
        p_biasesGradients[get_global_id(0)] /= p_batchSize;
}
