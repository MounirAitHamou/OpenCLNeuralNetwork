#include "NeuralNetwork/NeuralNetwork.hpp"

cl::Buffer NeuralNetwork::forward(const cl::Buffer& initial_input_batch, size_t current_batch_actual_size) {
    cl::Buffer current_input_buffer = initial_input_batch;

    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        layer->setBatchSize(current_batch_actual_size);
        layer->runForward(current_input_buffer);
        current_input_buffer = layer->outputs;
    }
    return current_input_buffer;
}

void NeuralNetwork::backprop(const cl::Buffer& original_network_input_batch, const cl::Buffer& target_batch_buf) {
    size_t output_size = layers.back()->output_dims.getTotalElements();

    auto& output_layer = layers.back();

    // --- Step 1: Compute Deltas for the Output Layer ---
    // This is the starting point of backpropagation. It calculates the initial
    // error signal based on the difference between network output and true targets.
    output_layer->computeOutputDeltas(target_batch_buf, loss_function_type);

    // --- Step 2: Backpropagate Deltas through Hidden Layers ---
    // Iterate from the second-to-last layer down to the first hidden layer.
    // The loop variable 'l' represents the index of the current layer being processed.
    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
        auto& current_layer = layers[l];
        auto& next_layer = layers[l + 1];
        current_layer->backpropDeltas(next_layer->weights, next_layer->deltas, next_layer->output_dims.getTotalElements());
    }

    // --- Step 3: Calculate Weight and Bias Gradients for All Layers ---
    // Iterate from the output layer back to the input layer.
    // Gradients depend on the inputs to the current layer and its deltas.
    for (int l = static_cast<int>(layers.size()) - 1; l >= 0; --l) {
        auto& layer = layers[l];

        // If it's the first layer (l == 0), its input is the original network input batch.
        // Otherwise, its input is the output of the previous layer (layers[l - 1]->outputs).
        const cl::Buffer& inputs_to_current_layer = (l == 0) ? original_network_input_batch : layers[l - 1]->outputs;

        layer->calculateWeightGradients(inputs_to_current_layer);

        layer->calculateBiasGradients();
    }

    // --- Step 4: Update Parameters (Weights and Biases) using the Optimizer ---
    // Iterate through all layers to update their weights and biases.
    for (auto& layer : layers) {
        optimizer->updateParameters(layer->weights, layer->weight_gradients, layer->getWeightsSize());
        optimizer->updateParameters(layer->biases, layer->bias_gradients, layer->getBiasesSize());
    }

    // --- Step 5: Perform Optimizer's Global Step (e.g., update Adam's 't' counter) ---
    optimizer->step();
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs,
                          const std::vector<std::vector<float>>& targets,
                          int epochs) {

    if (inputs.size() != targets.size()) {
        std::cerr << "Error: Input and target datasets must have the same number of samples." << std::endl;
        return;
    }

    size_t input_single_size = inputs[0].size();
    size_t output_single_size = targets[0].size();
    size_t num_samples = inputs.size(); 

    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<float> current_batch_input_host(batch_size * input_single_size);
    std::vector<float> current_batch_target_host(batch_size * output_single_size);
    std::vector<float> network_output_host(batch_size * output_single_size);

    cl::Buffer current_batch_input_buf(context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * input_single_size);
    cl::Buffer current_batch_target_buf(context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * output_single_size);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

        for (size_t i = 0; i < num_samples; i += batch_size) {
            size_t current_batch_actual_size = std::min((size_t)batch_size, num_samples - i);

            for (size_t j = 0; j < current_batch_actual_size; ++j) {
                std::copy(inputs[indices[i + j]].begin(), inputs[indices[i + j]].end(),
                          current_batch_input_host.begin() + j * input_single_size);
                std::copy(targets[indices[i + j]].begin(), targets[indices[i + j]].end(),
                          current_batch_target_host.begin() + j * output_single_size);
            }

            queue.enqueueWriteBuffer(current_batch_input_buf, CL_TRUE, 0,
                                     sizeof(float) * current_batch_actual_size * input_single_size,
                                     current_batch_input_host.data());
            queue.enqueueWriteBuffer(current_batch_target_buf, CL_TRUE, 0,
                                     sizeof(float) * current_batch_actual_size * output_single_size,
                                     current_batch_target_host.data());

            cl::Buffer final_output_buf = forward(current_batch_input_buf, current_batch_actual_size);

            queue.enqueueReadBuffer(final_output_buf, CL_TRUE, 0,
                                    sizeof(float) * current_batch_actual_size * output_single_size,
                                    network_output_host.data());

            for (size_t b = 0; b < current_batch_actual_size; ++b) {
                for (size_t j = 0; j < output_single_size; ++j) {
                    float diff = network_output_host[b * output_single_size + j] -
                                 current_batch_target_host[b * output_single_size + j];
                    total_loss += diff * diff;
                }
            }

            backprop(current_batch_input_buf, current_batch_target_buf);

            queue.finish();
        }

        float average_loss = total_loss / (num_samples * output_single_size);
        std::cout << "Epoch " << (epoch + 1) << " | Loss: " << average_loss << std::endl;
    }
}