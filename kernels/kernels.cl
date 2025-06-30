float activate(float x) {
    return 1.0f / (1.0f + exp(-x));        // Sigmoid 
}

float activate_derivative(float x) {
    return x * (1.0f - x);
}

__kernel void layer_forward_batch(
    __global const float* inputs,          // [batch_size * input_size]
    __global const float* weights,         // [output_size * input_size]
    __global const float* biases,          // [output_size]
    __global float* outputs,               // [batch_size * output_size]
    const int input_size,
    const int output_size,
    const int batch_size)
{
    int global_id = get_global_id(0);      //Output neuron index
    int batch_idx = get_global_id(1);      //Batch sample index

    if (global_id >= output_size || batch_idx >= batch_size) return;

    float sum = biases[global_id];

    for (int i = 0; i < input_size; ++i) {
        sum += weights[global_id * input_size + i] * inputs[batch_idx * input_size + i];
    }

    outputs[batch_idx * output_size + global_id] = activate(sum);
}

__kernel void compute_output_delta_batch(
    __global const float* outputs,         // [batch_size * output_size]
    __global const float* targets,         // [batch_size * output_size]
    __global float* deltas,                // [batch_size * output_size]
    const int output_size,
    const int batch_size)
{
    int global_id = get_global_id(0);
    int batch_idx = get_global_id(1);
    if (global_id >= output_size || batch_idx >= batch_size) return;
    float output_value = outputs[batch_idx * output_size + global_id];
    deltas[batch_idx * output_size + global_id] = 2.0f * (output_value - targets[batch_idx * output_size + global_id]) * activate_derivative(output_value);
}

__kernel void backpropagate_delta_batch(
    __global const float* next_weights,    // [next_output_size * current_output_size]
    __global const float* next_deltas,     // [batch_size * next_output_size]
    __global float* current_deltas,        // [batch_size * current_output_size]
    __global const float* current_outputs, // [batch_size * current_output_size] (for derivative)
    const int next_output_size,
    const int current_output_size,
    const int batch_size)
{
    int global_id = get_global_id(0);
    int batch_idx = get_global_id(1);
    if (global_id >= current_output_size || batch_idx >= batch_size) return;
    float sum_weighted_deltas = 0.0f;
    for (int j = 0; j < next_output_size; ++j) {
        sum_weighted_deltas += next_weights[j * current_output_size + global_id] * next_deltas[batch_idx * next_output_size + j];
    }
    current_deltas[batch_idx * current_output_size + global_id] = sum_weighted_deltas * activate_derivative(current_outputs[batch_idx * current_output_size + global_id]);
}

__kernel void update_weights_batch(
    __global float* weights,               // [output_size * input_size]
    __global float* biases,                // [output_size]
    __global const float* input,           // [batch_size * input_size]
    __global const float* deltas,          // [batch_size * output_size]
    const int input_size,
    const int output_size,
    const float learning_rate,
    const int batch_size)
{
    int out_neuron_idx = get_global_id(0); // Output neuron index
    int in_neuron_idx = get_global_id(1);  // Input neuron index
    if (out_neuron_idx >= output_size || in_neuron_idx >= input_size) return;
    float weight_delta_sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        weight_delta_sum += deltas[b * output_size + out_neuron_idx] * input[b * input_size + in_neuron_idx];
    }
    weights[out_neuron_idx * input_size + in_neuron_idx] -= learning_rate * weight_delta_sum / batch_size;
    if (in_neuron_idx == 0) {
        float bias_delta_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            bias_delta_sum += deltas[b * output_size + out_neuron_idx];
        }
        biases[out_neuron_idx] -= learning_rate * bias_delta_sum / batch_size;
    }
}