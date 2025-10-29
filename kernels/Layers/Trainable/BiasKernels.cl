__kernel void convolutionalBias(
    __global const float* p_biases,
    __global float* p_outputs,
    const unsigned int p_totalOutputElements,
    const unsigned int p_outputChannels)
{
    const unsigned int globalId = get_global_id(0);
    const unsigned int batchIndex = get_global_id(1);

    const unsigned int batchOffset = batchIndex * p_totalOutputElements;
    const unsigned int outputIndex = batchOffset + globalId;

    const unsigned int spatialElementsPerImage = p_totalOutputElements / p_outputChannels;
    const unsigned int outputChannelIndex = globalId / spatialElementsPerImage;

    p_outputs[outputIndex] += p_biases[outputChannelIndex];
}

__kernel void denseBias(
    __global const float* p_biases,
    __global float* p_outputs,
    const unsigned int p_outputSize) {
    unsigned int outNeuronIdx = get_global_id(0);
    unsigned int idx = get_global_id(1) * p_outputSize + outNeuronIdx;
    p_outputs[idx] += p_biases[outNeuronIdx];
}
