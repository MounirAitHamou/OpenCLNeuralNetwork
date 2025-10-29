__kernel void sgdUpdateParameters(
    __global float* p_parameters,
    __global const float* p_gradients,
    const float p_learningRate,
    const float p_weightDecayRate) {

    unsigned int idx = get_global_id(0);
    
    p_parameters[idx] -= p_learningRate * (p_gradients[idx] + p_weightDecayRate * p_parameters[idx]);
}
