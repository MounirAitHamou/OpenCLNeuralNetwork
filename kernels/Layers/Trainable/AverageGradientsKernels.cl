__kernel void averageWeightsGradients(
    __global float* p_weightsGradients,
    const unsigned int p_batchSize) {
        p_weightsGradients[get_global_id(0)] /= p_batchSize;
}

__kernel void averageBiasesGradients(
    __global float* p_biasesGradients,
    const unsigned int p_batchSize) {
        p_biasesGradients[get_global_id(0)] /= p_batchSize;
}
