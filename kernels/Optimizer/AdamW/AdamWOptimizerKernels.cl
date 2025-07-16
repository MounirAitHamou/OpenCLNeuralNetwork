__kernel void adamw_update_parameters(
    __global float* params,
    __global float* grads,
    __global float* m,
    __global float* v,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int num_elements,
    float weight_decay_rate,
    float beta1_pow_t,
    float beta2_pow_t
) {
    int gid = get_global_id(0);

    if (gid >= num_elements) {
        return;
    }

    float current_gradient = grads[gid];

    m[gid] = beta1 * m[gid] + (1.0f - beta1) * current_gradient;

    v[gid] = beta2 * v[gid] + (1.0f - beta2) * current_gradient * current_gradient;

    float m_hat = m[gid] / (1.0f - beta1_pow_t);
    float v_hat = v[gid] / (1.0f - beta2_pow_t);

    float adam_update_term = learning_rate * m_hat / (sqrt(v_hat) + epsilon);

    params[gid] -= (learning_rate * weight_decay_rate * params[gid]);

    params[gid] -= adam_update_term;
}