/**
 * @kernel sgd_update_parameters
 * @brief OpenCL kernel for updating neural network parameters (weights or biases)
 * using the Stochastic Gradient Descent (SGD) optimization algorithm.
 *
 * This kernel applies the SGD update rule for each parameter element in parallel.
 * It subtracts the scaled gradient from the parameter, and also incorporates
 * L2 regularization (weight decay).
 *
 * @param params_buf Global memory buffer containing the parameters (weights or biases) to be updated.
 * These values will be modified in place.
 * @param grads_buf Global memory buffer containing the gradients for the corresponding parameters.
 * These are the gradients of the loss with respect to the parameters.
 * @param num_elements The total number of float elements in the parameter and gradient buffers.
 * @param learning_rate The global learning rate for the optimizer. This scalar value
 * controls the step size in the direction of the negative gradient.
 * @param weight_decay_rate The rate for L2 regularization (weight decay). This term
 * is added to the gradient to penalize large weights.
 */
__kernel void sgd_update_parameters(
    __global float* params_buf,
    __global const float* grads_buf,
    const unsigned int num_elements,
    const float learning_rate,
    const float weight_decay_rate) {

    // Get the global ID of the current work-item.
    // Each work-item is responsible for updating one parameter element.
    unsigned int gid = get_global_id(0);

    // Boundary check: Ensure the global ID is within the valid range of elements.
    // This prevents out-of-bounds memory access if the global work size is larger than num_elements.
    if (gid >= num_elements) return;

    // Retrieve the current gradient for this parameter element.
    float current_gradient = grads_buf[gid];

    // --- Apply Weight Decay (L2 Regularization) ---
    // The weight decay term (weight_decay_rate * params_buf[gid]) is added to the gradient.
    // This effectively pushes the weights towards zero, helping to prevent overfitting.
    // The combined term (current_gradient + weight_decay_rate * params_buf[gid])
    // represents the total direction of update for the parameter.

    // --- Update Parameter ---
    // The core SGD update rule:
    // param = param - learning_rate * (gradient + weight_decay_rate * param)
    // The learning_rate scales the magnitude of the update step.
    params_buf[gid] -= learning_rate * (current_gradient + weight_decay_rate * params_buf[gid]);
}