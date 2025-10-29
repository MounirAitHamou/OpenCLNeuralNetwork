__kernel void adamWUpdateParameters(
    __global float* p_parameters,
    __global const float* p_gradients,
    __global float* p_m,
    __global float* p_v,
    float p_learningRate,
    float p_beta1,
    float p_beta2,
    float p_epsilon,
    float p_weightDecayRate,
    float p_beta1PowT,
    float p_beta2PowT
) {
    int idx = get_global_id(0);

    float currentGradient = p_gradients[idx];
    
    float m_t = p_beta1 * p_m[idx] + (1.0f - p_beta1) * currentGradient;
    float v_t = p_beta2 * p_v[idx] + (1.0f - p_beta2) * currentGradient * currentGradient;
    
    p_m[idx] = m_t;
    p_v[idx] = v_t;

    float m_hat = m_t / (1.0f - p_beta1PowT);
    float v_hat = v_t / (1.0f - p_beta2PowT);

    p_parameters[idx] -= p_learningRate * (p_weightDecayRate * p_parameters[idx] + (m_hat / (sqrt(v_hat) + p_epsilon)));
}
