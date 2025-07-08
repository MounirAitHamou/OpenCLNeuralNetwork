// --- Activation Function Type Definitions ---
// These preprocessor macros define integer codes for different activation functions.
// They are typically used to pass activation type information to OpenCL kernels,
// allowing a single kernel to handle multiple activation functions via a switch statement.

/**
 * @def ACT_LINEAR
 * @brief Represents the Linear activation function.
 * Output is equal to the input (f(x) = x).
 */
#define ACT_LINEAR    0

/**
 * @def ACT_RELU
 * @brief Represents the Rectified Linear Unit (ReLU) activation function.
 * Output is max(0, x).
 */
#define ACT_RELU      1

/**
 * @def ACT_SIGMOID
 * @brief Represents the Sigmoid activation function.
 * Output is 1 / (1 + exp(-x)).
 */
#define ACT_SIGMOID   2

/**
 * @def ACT_TANH
 * @brief Represents the Hyperbolic Tangent (Tanh) activation function.
 * Output is tanh(x).
 */
#define ACT_TANH      3

// --- Loss Function Type Definitions ---
// These preprocessor macros define integer codes for different loss functions.
// Similar to activation types, they can be used to select the appropriate
// loss calculation logic within a generic OpenCL kernel.

/**
 * @def LOSS_MSE
 * @brief Represents the Mean Squared Error (MSE) loss function.
 */
#define LOSS_MSE      0

/**
 * @def LOSS_BCE
 * @brief Represents the Binary Cross-Entropy (BCE) loss function.
 */
#define LOSS_BCE      1

/**
 * @brief Applies a specified activation function to a given float value.
 *
 * This function acts as a dispatcher, selecting and applying the correct
 * activation function based on the provided `activation_type` code.
 * It is designed for use within OpenCL kernels or CPU-side utility functions.
 *
 * @param x The input float value to which the activation function will be applied.
 * @param activation_type An unsigned integer representing the type of activation
 * function to apply (e.g., ACT_LINEAR, ACT_RELU, etc.).
 * @return The result of applying the specified activation function to `x`.
 */
float apply_activation(float x, unsigned int activation_type) {
    // Use a switch statement to select the appropriate activation function logic.
    switch (activation_type) {
        case ACT_LINEAR:
            // For linear activation, the output is simply the input.
            return x;
        case ACT_RELU:
            // For ReLU, return the maximum of 0.0f and the input x.
            return fmax(0.0f, x);
        case ACT_SIGMOID:
            // For Sigmoid, calculate 1.0f / (1.0f + e^(-x)).
            // `exp` is the exponential function (e^x).
            return 1.0f / (1.0f + exp(-x));
        case ACT_TANH:
            // For Tanh, use the standard hyperbolic tangent function.
            return tanh(x);
        default:
            // If an unknown activation_type is provided, default to linear activation
            // or handle as an error, depending on desired behavior.
            // Here, it defaults to linear.
            return x;
    }
}

/**
 * @brief Computes the derivative of a specified activation function.
 *
 * This function calculates the derivative of the activation function with respect to its input `x`.
 * The derivative is crucial during the backpropagation process to determine how much
 * the error changes with respect to the pre-activation values.
 *
 * @param x The input value to the activation function (typically the pre-activation value).
 * @param activation_type An unsigned integer representing the type of activation
 * function whose derivative is to be computed (e.g., ACT_LINEAR, ACT_RELU, etc.).
 * @return The derivative of the activation function at point `x`.
 */
float apply_activation_derivative(float x, unsigned int activation_type) {
    switch (activation_type) {
        case ACT_LINEAR:
            // Derivative of f(x) = x is f'(x) = 1.0.
            return 1.0f;
        case ACT_RELU:
            // Derivative of f(x) = max(0, x) is 1.0 if x > 0, and 0.0 if x <= 0.
            // (Note: At x=0, the derivative is undefined, but typically 0 or 1 is chosen).
            return (x > 0.0f) ? 1.0f : 0.0f;
        case ACT_SIGMOID: {
            // Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x)).
            // We first compute sigmoid(x) to use it in the derivative formula.
            float sigmoid_x = 1.0f / (1.0f + exp(-x));
            return sigmoid_x * (1.0f - sigmoid_x);
        }
        case ACT_TANH: {
            // Derivative of tanh(x) is 1 - tanh(x)^2.
            // We first compute tanh(x) to use it in the derivative formula.
            float tanh_x = tanh(x);
            return 1.0f - tanh_x * tanh_x;
        }
        default:
            // If an unknown activation_type is provided, default to linear derivative.
            return 1.0f;
    }
}

/**
 * @brief Computes the derivative of a specified loss function.
 *
 * This function calculates the derivative of the loss function with respect to the network's
 * predicted output. This is the first step in backpropagation for the output layer.
 *
 * @param pred The predicted output value from the network.
 * @param target The true target (label) value.
 * @param loss_type An unsigned integer representing the type of loss function
 * whose derivative is to be computed (e.g., LOSS_MSE, LOSS_BCE).
 * @return The derivative of the loss function.
 */
float compute_loss_derivative(float pred, float target, unsigned int loss_type) {
    switch (loss_type) {
        case LOSS_MSE:
            // Derivative of MSE (0.5 * (pred - target)^2) with respect to pred is (pred - target).
            return (pred - target);
        case LOSS_BCE:
            // Derivative of Binary Cross-Entropy loss with respect to pred.
            // A small epsilon (1e-7f) is added to the denominator for numerical stability
            // to prevent division by zero when pred is exactly 0.0 or 1.0.
            return (pred - target) / (pred * (1.0f - pred) + 1e-7f);
        default:
            // If an unknown loss_type is provided, return 0.0.
            return 0.0f;
    }
}

/**
 * @kernel dense_forward_batch
 * @brief OpenCL kernel for performing the forward pass of a dense (fully connected) layer in batch.
 *
 * This kernel calculates the pre-activation values (weighted sum + bias) and then applies
 * the specified activation function to produce the layer's output for each neuron
 * in each sample of the batch.
 *
 * @param weights_buf Global memory buffer containing the layer's weights.
 * Layout: [output_size * input_size] (row-major for output neuron, column-major for input neuron).
 * @param biases_buf Global memory buffer containing the layer's biases.
 * Layout: [output_size].
 * @param input_buf Global memory buffer containing the input batch from the previous layer.
 * Layout: [batch_size * input_size].
 * @param output_buf Global memory buffer to store the activated outputs of the current layer.
 * Layout: [batch_size * output_size].
 * @param pre_activation_buf Global memory buffer to store the pre-activation values (sums before activation).
 * Layout: [batch_size * output_size].
 * @param input_size The number of neurons in the input to this layer.
 * @param output_size The number of neurons in the output of this layer.
 * @param batch_size The number of samples in the current batch.
 * @param activation_type An integer code specifying the activation function to apply (e.g., ACT_RELU).
 */
__kernel void dense_forward_batch(
    __global const float* weights_buf,
    __global const float* biases_buf,
    __global const float* input_buf,
    __global float* output_buf,
    __global float* pre_activation_buf,
    const unsigned int input_size,
    const unsigned int output_size,
    const unsigned int batch_size,
    const unsigned int activation_type) {

    // Get the global ID for the output neuron index (first dimension of global work size).
    unsigned int out_neuron_idx = get_global_id(0);
    // Get the global ID for the batch index (second dimension of global work size).
    unsigned int batch_idx = get_global_id(1);

    // Boundary check: Ensure the current work-item is within the valid range of output neurons and batch samples.
    if (out_neuron_idx >= output_size || batch_idx >= batch_size) return;

    // Initialize sum with the bias for the current output neuron.
    float sum = biases_buf[out_neuron_idx];

    // Perform the dot product (weighted sum) of inputs and weights for the current output neuron.
    // weights_buf[out_neuron_idx * input_size + i] accesses the weight connecting input neuron 'i' to 'out_neuron_idx'.
    // input_buf[batch_idx * input_size + i] accesses the input from neuron 'i' for the current 'batch_idx'.
    for (unsigned int i = 0; i < input_size; ++i) {
        sum += weights_buf[out_neuron_idx * input_size + i] * input_buf[batch_idx * input_size + i];
    }

    // Store the calculated pre-activation sum. This is needed for backpropagation.
    pre_activation_buf[batch_idx * output_size + out_neuron_idx] = sum;

    // Apply the specified activation function to the sum and store the result as the layer's output.
    output_buf[batch_idx * output_size + out_neuron_idx] = apply_activation(sum, activation_type);
}

/**
 * @kernel dense_compute_output_deltas_batch
 * @brief OpenCL kernel for computing the error deltas for the output layer of a dense network in batch.
 *
 * This kernel calculates the initial deltas (error gradients) for each output neuron
 * in each sample of the batch. It combines the derivative of the loss function
 * with the derivative of the activation function.
 *
 * @param pre_activation_buf Global memory buffer containing the pre-activation values of the output layer.
 * @param output_buf Global memory buffer containing the activated outputs of the output layer.
 * @param target_buf Global memory buffer containing the true target (label) values for the batch.
 * @param deltas_buf Global memory buffer to store the computed deltas for the output layer.
 * Layout: [batch_size * output_size].
 * @param output_size The number of neurons in the output layer.
 * @param batch_size The number of samples in the current batch.
 * @param activation_type An integer code specifying the activation function used by the output layer.
 * @param loss_function_type An integer code specifying the loss function used (e.g., LOSS_MSE, LOSS_BCE).
 */
__kernel void dense_compute_output_deltas_batch(
    __global const float* pre_activation_buf,
    __global const float* output_buf,
    __global const float* target_buf,
    __global float* deltas_buf,
    const unsigned int output_size,
    const unsigned int batch_size,
    const unsigned int activation_type,
    const unsigned int loss_function_type
    ) {

    // Get the global ID for the output neuron index.
    unsigned int out_neuron_idx = get_global_id(0);
    // Get the global ID for the batch index.
    unsigned int batch_idx = get_global_id(1);

    // Boundary check: Ensure the current work-item is within valid bounds.
    if (out_neuron_idx >= output_size || batch_idx >= batch_size) return;

    // Calculate the delta for the current output neuron and batch sample.
    // Delta = (Derivative of Loss w.r.t. prediction) * (Derivative of Activation w.r.t. pre-activation).
    deltas_buf[batch_idx * output_size + out_neuron_idx] =
        compute_loss_derivative(output_buf[batch_idx * output_size + out_neuron_idx], target_buf[batch_idx * output_size + out_neuron_idx], loss_function_type) *
        apply_activation_derivative(pre_activation_buf[batch_idx * output_size + out_neuron_idx], activation_type);
}

/**
 * @kernel dense_backprop_deltas_batch
 * @brief OpenCL kernel for backpropagating error deltas through a dense layer in batch.
 *
 * This kernel calculates the deltas for the current layer based on the deltas
 * of the subsequent layer and the weights connecting them. This is a crucial step
 * in the backpropagation algorithm for hidden layers.
 *
 * @param next_layer_weights_buf Global memory buffer containing the weights of the *next* layer.
 * Layout: [next_layer_output_size * current_layer_output_size].
 * @param next_layer_deltas_buf Global memory buffer containing the deltas of the *next* layer.
 * Layout: [batch_size * next_layer_output_size].
 * @param current_layer_deltas_buf Global memory buffer to store the computed deltas for the *current* layer.
 * Layout: [batch_size * current_layer_output_size].
 * @param current_layer_pre_activation_buf Global memory buffer containing the pre-activation values of the current layer.
 * @param current_layer_output_size The number of neurons in the output of the current layer.
 * @param next_layer_output_size The number of neurons in the output of the next layer.
 * @param batch_size The number of samples in the current batch.
 * @param activation_type An integer code specifying the activation function used by the current layer.
 */
__kernel void dense_backprop_deltas_batch(
    __global const float* next_layer_weights_buf,
    __global const float* next_layer_deltas_buf,
    __global float* current_layer_deltas_buf,
    __global const float* current_layer_pre_activation_buf,
    const unsigned int current_layer_output_size,
    const unsigned int next_layer_output_size,
    const unsigned int batch_size,
    const unsigned int activation_type) {

    // Get the global ID for the current neuron index in this layer.
    unsigned int current_neuron_idx = get_global_id(0);
    // Get the global ID for the batch index.
    unsigned int batch_idx = get_global_id(1);

    // Boundary check: Ensure the current work-item is within valid bounds.
    if (current_neuron_idx >= current_layer_output_size || batch_idx >= batch_size) return;

    // Sum the error contributions from all neurons in the next layer.
    // This is a weighted sum of the next layer's deltas, where weights are
    // the connections from the current neuron to the next layer's neurons.
    float error_sum_from_next_layer = 0.0f;
    for (unsigned int k = 0; k < next_layer_output_size; ++k) {
        // next_layer_weights_buf[k * current_layer_output_size + current_neuron_idx]
        // accesses the weight connecting 'current_neuron_idx' in the current layer
        // to neuron 'k' in the next layer.
        error_sum_from_next_layer +=
            next_layer_weights_buf[k * current_layer_output_size + current_neuron_idx] *
            next_layer_deltas_buf[batch_idx * next_layer_output_size + k];
    }

    // Calculate the delta for the current neuron and batch sample.
    // This is the error sum from the next layer multiplied by the derivative
    // of the current layer's activation function (at its pre-activation value).
    current_layer_deltas_buf[batch_idx * current_layer_output_size + current_neuron_idx] =
        error_sum_from_next_layer * apply_activation_derivative(current_layer_pre_activation_buf[batch_idx * current_layer_output_size + current_neuron_idx], activation_type);
}

/**
 * @kernel dense_calculate_weight_gradients_batch
 * @brief OpenCL kernel for calculating weight gradients for a dense layer in batch.
 *
 * This kernel computes the gradients for each weight in the dense layer by
 * averaging the product of input activations and error deltas across the batch.
 *
 * @param inputs_buf Global memory buffer containing the input activations to the current layer.
 * Layout: [batch_size * input_size].
 * @param deltas_buf Global memory buffer containing the computed deltas for the current layer.
 * Layout: [batch_size * output_size].
 * @param weight_gradients_buf Global memory buffer to store the calculated weight gradients.
 * Layout: [output_size * input_size].
 * @param input_size The number of neurons in the input to this layer.
 * @param output_size The number of neurons in the output of this layer.
 * @param batch_size The number of samples in the current batch.
 */
__kernel void dense_calculate_weight_gradients_batch(
    __global const float* inputs_buf,
    __global const float* deltas_buf,
    __global float* weight_gradients_buf,
    const unsigned int input_size,
    const unsigned int output_size,
    const unsigned int batch_size) {

    // Get the global ID for the output neuron index.
    unsigned int out_neuron_idx = get_global_id(0);
    // Get the global ID for the input neuron index.
    unsigned int in_neuron_idx = get_global_id(1);

    // Boundary check: Ensure the current work-item is within valid bounds.
    if (out_neuron_idx >= output_size || in_neuron_idx >= input_size) return;

    // Initialize a sum for the gradient over the batch.
    float gradient_sum = 0.0f;

    // Iterate over all samples in the batch to accumulate the gradient.
    // The gradient for a weight (from input_i to output_j) is delta_j * input_i.
    // We sum this product over all samples in the batch.
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        gradient_sum += deltas_buf[batch_idx * output_size + out_neuron_idx] *
                        inputs_buf[batch_idx * input_size + in_neuron_idx];
    }

    // Store the average gradient for the current weight.
    // The sum is divided by the batch_size to get the average gradient.
    weight_gradients_buf[out_neuron_idx * input_size + in_neuron_idx] = gradient_sum / batch_size;
}

/**
 * @kernel dense_calculate_bias_gradients_batch
 * @brief OpenCL kernel for calculating bias gradients for a dense layer in batch.
 *
 * This kernel computes the gradients for each bias in the dense layer by
 * averaging the error deltas for that bias across the batch.
 *
 * @param deltas_buf Global memory buffer containing the computed deltas for the current layer.
 * Layout: [batch_size * output_size].
 * @param bias_gradients_buf Global memory buffer to store the calculated bias gradients.
 * Layout: [output_size].
 * @param output_size The number of neurons in the output of this layer.
 * @param batch_size The number of samples in the current batch.
 */
__kernel void dense_calculate_bias_gradients_batch(
    __global const float* deltas_buf,
    __global float* bias_gradients_buf,
    const unsigned int output_size,
    const unsigned int batch_size) {

    // Get the global ID for the output neuron index (which corresponds to a bias).
    unsigned int out_neuron_idx = get_global_id(0);

    // Boundary check: Ensure the current work-item is within valid bounds.
    if (out_neuron_idx >= output_size) return;

    // Initialize a sum for the gradient over the batch.
    float gradient_sum = 0.0f;

    // Iterate over all samples in the batch to accumulate the gradient for the current bias.
    // The gradient for a bias is simply the delta for that neuron.
    // We sum this delta over all samples in the batch.
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        gradient_sum += deltas_buf[batch_idx * output_size + out_neuron_idx];
    }

    // Store the average gradient for the current bias.
    // The sum is divided by the batch_size to get the average gradient.
    bias_gradients_buf[out_neuron_idx] = gradient_sum / batch_size;
}