__kernel void averageGradients(
    __global float* p_weightsGradients,
    const unsigned int p_batchSize) {
        p_weightsGradients[get_global_id(0)] /= p_batchSize;
}