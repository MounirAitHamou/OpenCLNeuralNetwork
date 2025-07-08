/**
 * @kernel adam_update_parameters
 * @brief OpenCL kernel for updating neural network parameters (weights or biases) using the Adam optimization algorithm.
 *
 * This kernel applies the Adam update rule for each parameter element in parallel.
 * It incorporates bias correction for the first and second moment estimates,
 * and also includes decoupled weight decay.
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
 * @param weight_decay_rate The rate for L2 regularization (weight decay). This is added to the gradient.
 * @param beta1_pow_t The beta1 parameter raised to the power of the current time step (t), used for bias correction of m.
 * @param beta2_pow_t The beta2 parameter raised to the power of the current time step (t), used for bias correction of v.
 */
__kernel void adam_update_parameters(
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
    if (gid >= num_elements) return;

    // --- 1. Apply Weight Decay (L2 Regularization) ---
    // Add the weight decay term to the current gradient. This is common in Adam.
    // The term is `weight_decay_rate * params[gid]`.
    float current_gradient = grads[gid];
    current_gradient += weight_decay_rate * params[gid];

    // --- 2. Update Biased First Moment Estimate (m) ---
    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    // This is an exponential moving average of the current and past gradients.
    m[gid] = beta1 * m[gid] + (1.0f - beta1) * current_gradient;

    // --- 3. Update Biased Second Moment Estimate (v) ---
    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    // This is an exponential moving average of the current and past squared gradients.
    v[gid] = beta2 * v[gid] + (1.0f - beta2) * current_gradient * current_gradient;

    // --- 4. Apply Bias Correction to Moments and Update Parameters---
    // m_hat = m_t / (1 - beta1_pow_t)
    // v_hat = v_t / (1 - beta2_pow_t)
    // These corrections account for the fact that the moment estimates are initialized at zero,
    // which biases them towards zero, especially in early training steps.
    // params_t = params_{t-1} - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    // The learning rate is scaled by the bias-corrected first moment and the
    // square root of the bias-corrected second moment (plus epsilon for stability).
    params[gid] -= (learning_rate * (m[gid] / (1.0f - beta1_pow_t))) / (sqrt((v[gid] / (1.0f - beta2_pow_t))) + epsilon);
}