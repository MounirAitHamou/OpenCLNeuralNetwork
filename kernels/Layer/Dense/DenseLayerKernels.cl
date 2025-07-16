#define ACT_LINEAR    0
#define ACT_RELU      1
#define ACT_SIGMOID   2
#define ACT_TANH      3

#define LOSS_MSE      0
#define LOSS_BCE      1

float apply_activation(float x, unsigned int activation_type) {
    switch (activation_type) {
        case ACT_LINEAR:
            return x;
        case ACT_RELU:
            return fmax(0.0f, x);
        case ACT_SIGMOID:
            return 1.0f / (1.0f + exp(-x));
        case ACT_TANH:
            return tanh(x);
        default:
            return x;
    }
}

float apply_activation_derivative(float x, unsigned int activation_type) {
    switch (activation_type) {
        case ACT_LINEAR:
            return 1.0f;
        case ACT_RELU:
            return (x > 0.0f) ? 1.0f : 0.0f;
        case ACT_SIGMOID: {
            float sigmoid_x = 1.0f / (1.0f + exp(-x));
            return sigmoid_x * (1.0f - sigmoid_x);
        }
        case ACT_TANH: {
            float tanh_x = tanh(x);
            return 1.0f - tanh_x * tanh_x;
        }
        default:
            return 1.0f;
    }
}

float compute_loss_derivative(float pred, float target, unsigned int loss_type) {
    switch (loss_type) {
        case LOSS_MSE:
            return (pred - target);
        case LOSS_BCE:
            return (pred - target) / (pred * (1.0f - pred) + 1e-7f);
        default:
            return 0.0f;
    }
}

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

    unsigned int out_neuron_idx = get_global_id(0);
    unsigned int batch_idx = get_global_id(1);

    if (out_neuron_idx >= output_size || batch_idx >= batch_size) return;

    float sum = biases_buf[out_neuron_idx];

    for (unsigned int i = 0; i < input_size; ++i) {
        sum += weights_buf[out_neuron_idx * input_size + i] * input_buf[batch_idx * input_size + i];
    }

    pre_activation_buf[batch_idx * output_size + out_neuron_idx] = sum;

    output_buf[batch_idx * output_size + out_neuron_idx] = apply_activation(sum, activation_type);
}

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

    unsigned int out_neuron_idx = get_global_id(0);
    unsigned int batch_idx = get_global_id(1);

    if (out_neuron_idx >= output_size || batch_idx >= batch_size) return;

    deltas_buf[batch_idx * output_size + out_neuron_idx] =
        compute_loss_derivative(output_buf[batch_idx * output_size + out_neuron_idx], target_buf[batch_idx * output_size + out_neuron_idx], loss_function_type) *
        apply_activation_derivative(pre_activation_buf[batch_idx * output_size + out_neuron_idx], activation_type);
}

__kernel void dense_backprop_deltas_with_weights_batch(
    __global const float* next_layer_weights_buf,
    __global const float* next_layer_deltas_buf,
    __global float* current_layer_deltas_buf,
    __global const float* current_layer_pre_activation_buf,
    const unsigned int current_layer_output_size,
    const unsigned int next_layer_output_size,
    const unsigned int batch_size,
    const unsigned int activation_type) {

    unsigned int current_neuron_idx = get_global_id(0);
    unsigned int batch_idx = get_global_id(1);

    if (current_neuron_idx >= current_layer_output_size || batch_idx >= batch_size) return;

    float error_sum_from_next_layer = 0.0f;
    for (unsigned int k = 0; k < next_layer_output_size; ++k) {
        error_sum_from_next_layer +=
            next_layer_weights_buf[k * current_layer_output_size + current_neuron_idx] *
            next_layer_deltas_buf[batch_idx * next_layer_output_size + k];
    }

    current_layer_deltas_buf[batch_idx * current_layer_output_size + current_neuron_idx] =
        error_sum_from_next_layer * apply_activation_derivative(current_layer_pre_activation_buf[batch_idx * current_layer_output_size + current_neuron_idx], activation_type);
}


__kernel void dense_backprop_deltas_no_weights_batch(
    __global const float* downstream_deltas_buf,
    __global float* current_layer_deltas_buf,
    __global const float* current_layer_pre_activation_buf,
    const unsigned int current_layer_output_size,
    const unsigned int batch_size,
    const unsigned int activation_type) {

    unsigned int neuron_idx = get_global_id(0);
    unsigned int batch_idx = get_global_id(1);

    if (neuron_idx >= current_layer_output_size || batch_idx >= batch_size) return;

    unsigned int global_idx = batch_idx * current_layer_output_size + neuron_idx;

    float propagated_error = downstream_deltas_buf[global_idx];

    current_layer_deltas_buf[global_idx] =
        propagated_error * apply_activation_derivative(
            current_layer_pre_activation_buf[global_idx],
            activation_type
        );
}

__kernel void dense_calculate_weight_gradients_batch(
    __global const float* inputs_buf,
    __global const float* deltas_buf,
    __global float* weight_gradients_buf,
    const unsigned int input_size,
    const unsigned int output_size,
    const unsigned int batch_size) {

    unsigned int out_neuron_idx = get_global_id(0);
    unsigned int in_neuron_idx = get_global_id(1);

    if (out_neuron_idx >= output_size || in_neuron_idx >= input_size) return;

    float gradient_sum = 0.0f;

    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        gradient_sum += deltas_buf[batch_idx * output_size + out_neuron_idx] *
                        inputs_buf[batch_idx * input_size + in_neuron_idx];
    }

    weight_gradients_buf[out_neuron_idx * input_size + in_neuron_idx] = gradient_sum / batch_size;
}

__kernel void dense_calculate_bias_gradients_batch(
    __global const float* deltas_buf,
    __global float* bias_gradients_buf,
    const unsigned int output_size,
    const unsigned int batch_size) {

    unsigned int out_neuron_idx = get_global_id(0);

    if (out_neuron_idx >= output_size) return;

    float gradient_sum = 0.0f;

    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        gradient_sum += deltas_buf[batch_idx * output_size + out_neuron_idx];
    }

    bias_gradients_buf[out_neuron_idx] = gradient_sum / batch_size;
}