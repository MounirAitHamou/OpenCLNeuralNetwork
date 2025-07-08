/**
 * @kernel adamw_update_parameters
 * @brief OpenCL kernel for updating neural network parameters (weights or biases) using the AdamW optimization algorithm.
 *
 * AdamW (Adam with Weight Decay) decouples the weight decay from the gradient update,
 * applying it as a separate term. This often leads to better generalization performance
 * compared to standard Adam with L2 regularization directly applied to gradients.
 *
 * @param params Global memory buffer containing the parameters (weights or biases) to be updated.
 * These values will be modified in place.
 * @param grads Global memory buffer containing the gradients for the corresponding parameters.
 * @param m Global memory buffer for the first moment estimates (exponential moving average of gradients).
 * These values are updated in place.
 * @param v Global memory buffer for the second moment estimates (exponential moving average of squared gradients).
 * These values are updated in place.
 * @param learning_rate The global learning rate for the optimizer.
 * @param beta1 The exponential decay rate for the first moment estimates (m).
 * @param beta2 The exponential decay rate for the second moment estimates (v).
 * @param epsilon A small constant for numerical stability to prevent division by zero.
 * @param num_elements The total number of float elements in the parameter, gradient, and moment buffers.
 * @param weight_decay_rate The rate for decoupled L2 regularization (weight decay).
 * @param beta1_pow_t The beta1 parameter raised to the power of the current time step (t), used for bias correction of m.
 * @param beta2_pow_t The beta2 parameter raised to the power of the current time step (t), used for bias correction of v.
 */
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
    // Get the global ID of the current work-item. Each work-item processes one parameter element.
    int gid = get_global_id(0);

    // Boundary check: Ensure the work-item ID is within the valid range of elements.
    if (gid >= num_elements) {
        return;
    }

    // Get the current gradient for this parameter.
    float current_gradient = grads[gid];

    // --- 1. Update Biased First Moment Estimate (m) ---
    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    // This is an exponential moving average of the current and past gradients.
    m[gid] = beta1 * m[gid] + (1.0f - beta1) * current_gradient;

    // --- 2. Update Biased Second Moment Estimate (v) ---
    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    // This is an exponential moving average of the current and past squared gradients.
    v[gid] = beta2 * v[gid] + (1.0f - beta2) * current_gradient * current_gradient;

    // --- 3. Apply Bias Correction to Moments ---
    // m_hat = m_t / (1 - beta1_pow_t)
    // v_hat = v_t / (1 - beta2_pow_t)
    // These corrections account for the fact that the moment estimates are initialized at zero,
    // which biases them towards zero, especially in early training steps.
    float m_hat = m[gid] / (1.0f - beta1_pow_t);
    float v_hat = v[gid] / (1.0f - beta2_pow_t);

    // --- 4. Calculate Adam Update Term ---
    // This is the standard Adam update component before applying weight decay.
    // update_term = learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    float adam_update_term = learning_rate * m_hat / (sqrt(v_hat) + epsilon);

    // --- 5. Apply Decoupled Weight Decay ---
    // params_t = params_{t-1} - learning_rate * weight_decay_rate * params_{t-1}
    // This step applies L2 regularization directly to the parameters,
    // separate from the gradient-based update.
    params[gid] -= (learning_rate * weight_decay_rate * params[gid]);

    // --- 6. Apply Adam Update Term ---
    // params_t = params_t - adam_update_term
    // Finally, subtract the calculated Adam update term from the parameters.
    params[gid] -= adam_update_term;
}