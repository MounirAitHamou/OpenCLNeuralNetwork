__kernel void denseBias(
    __global const float* p_biases,
    __global float* p_outputs,
    const unsigned int p_outputSize) {
    unsigned int outNeuronIdx = get_global_id(0);
    unsigned int idx = get_global_id(1) * p_outputSize + outNeuronIdx;
    p_outputs[idx] += p_biases[outNeuronIdx];
}
