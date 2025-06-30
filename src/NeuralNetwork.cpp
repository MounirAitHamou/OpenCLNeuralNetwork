#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(OpenCLSetup ocl_setup, const int input_size, const std::vector<int>& hidden_layers_sizes, const int output_size, int batch_size, float learning_rate = 0.01f)
    : context(ocl_setup.context), queue(ocl_setup.queue), program(ocl_setup.program), learning_rate(learning_rate), batch_size(batch_size)
{
    int prev_size = input_size;

    for (int hidden_size : hidden_layers_sizes) {
        layers.emplace_back(ocl_setup, prev_size, hidden_size, batch_size);
        prev_size = hidden_size;
    }

    layers.emplace_back(ocl_setup, prev_size, output_size, batch_size);
}

void NeuralNetwork::initialize() {
    for (auto& layer : layers) {
        layer.setRandomParams();
    }
}

cl::Buffer NeuralNetwork::forward(const cl::Buffer& initial_input_batch) {
    cl::Buffer current_input_buffer = initial_input_batch;

    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        layer.batch_size = batch_size; 
        layer.runForward(current_input_buffer);
        current_input_buffer = layer.outputs;
    }
    return current_input_buffer;
}

void NeuralNetwork::backprop(const cl::Buffer& original_network_input_batch, const cl::Buffer& target_batch_buf) {
    int output_size = layers.back().output_size;

    cl::Kernel output_delta_kernel(program, "compute_output_delta_batch");
    auto& output_layer = layers.back();
    output_delta_kernel.setArg(0, output_layer.outputs);
    output_delta_kernel.setArg(1, target_batch_buf);
    output_delta_kernel.setArg(2, output_layer.deltas);
    output_delta_kernel.setArg(3, output_size);
    output_delta_kernel.setArg(4, batch_size);
    queue.enqueueNDRangeKernel(output_delta_kernel, cl::NullRange, cl::NDRange(output_size, batch_size), cl::NullRange);

    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
        auto& current_layer = layers[l];
        auto& next_layer = layers[l + 1];
        cl::Kernel backprop_kernel(program, "backpropagate_delta_batch");
        backprop_kernel.setArg(0, next_layer.weights);
        backprop_kernel.setArg(1, next_layer.deltas);
        backprop_kernel.setArg(2, current_layer.deltas);
        backprop_kernel.setArg(3, current_layer.outputs);
        backprop_kernel.setArg(4, next_layer.output_size);
        backprop_kernel.setArg(5, current_layer.output_size);
        backprop_kernel.setArg(6, batch_size);

        queue.enqueueNDRangeKernel(backprop_kernel, cl::NullRange, cl::NDRange(current_layer.output_size, batch_size), cl::NullRange);
    }

    for (int l = static_cast<int>(layers.size()) - 1; l >= 0; --l) {
        auto& layer = layers[l];
        cl::Buffer& inputs_to_current_layer = (l == 0) ? const_cast<cl::Buffer&>(original_network_input_batch) : layers[l - 1].outputs;

        cl::Kernel update_kernel(program, "update_weights_batch");
        update_kernel.setArg(0, layer.weights);
        update_kernel.setArg(1, layer.biases);
        update_kernel.setArg(2, inputs_to_current_layer);
        update_kernel.setArg(3, layer.deltas);
        update_kernel.setArg(4, layer.input_size);
        update_kernel.setArg(5, layer.output_size);
        update_kernel.setArg(6, learning_rate);
        update_kernel.setArg(7, batch_size);

        queue.enqueueNDRangeKernel(update_kernel, cl::NullRange, cl::NDRange(layer.output_size, layer.input_size), cl::NullRange);
    }
    queue.finish();
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs,
            const std::vector<std::vector<float>>& targets,
            int epochs)
{
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

            for (auto& layer : layers) {
                layer.batch_size = static_cast<int>(current_batch_actual_size);
            }

            cl::Buffer final_output_buf = forward(current_batch_input_buf);

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
        }

        float average_loss = total_loss / (num_samples * output_single_size);
        std::cout << "Epoch " << (epoch + 1) << " | Loss: " << average_loss << std::endl;
    }
}

bool NeuralNetwork::equalNetwork(const NeuralNetwork& other) const {
    if (learning_rate != other.learning_rate || batch_size != other.batch_size) {
        return false;
    }
    
    if (layers.size() != other.layers.size()) {
        return false;
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& this_layer = this->layers[i];
        const auto& other_layer = other.layers[i];
        
        if (this_layer.input_size != other_layer.input_size ||
            this_layer.output_size != other_layer.output_size) {
            return false;
        }
        
        std::vector<float> this_weights_host(this_layer.input_size * this_layer.output_size);
        std::vector<float> other_weights_host(other_layer.input_size * other_layer.output_size);
        queue.enqueueReadBuffer(this_layer.weights, CL_TRUE, 0, sizeof(float) * this_weights_host.size(), this_weights_host.data());
        queue.enqueueReadBuffer(other_layer.weights, CL_TRUE, 0, sizeof(float) * other_weights_host.size(), other_weights_host.data());

        std::vector<float> this_biases_host(this_layer.output_size);
        std::vector<float> other_biases_host(other_layer.output_size);
        queue.enqueueReadBuffer(this_layer.biases, CL_TRUE, 0, sizeof(float) * this_biases_host.size(), this_biases_host.data());
        queue.enqueueReadBuffer(other_layer.biases, CL_TRUE, 0, sizeof(float) * other_biases_host.size(), other_biases_host.data());

        queue.finish();

        if (this_weights_host != other_weights_host || this_biases_host != other_biases_host) {
            return false;
        }
    }
    return true;
}

int NeuralNetwork::save(const std::string& filename) const {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for saving: " << filename << std::endl;
        return 1;
    }

    outFile.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    outFile.write(reinterpret_cast<const char*>(&batch_size), sizeof(batch_size));
    int input_size_nn = layers.front().input_size;
    int output_size_nn = layers.back().output_size;
    outFile.write(reinterpret_cast<const char*>(&input_size_nn), sizeof(input_size_nn));
    outFile.write(reinterpret_cast<const char*>(&output_size_nn), sizeof(output_size_nn));

    size_t num_hidden_layers = layers.size() - 1;
    outFile.write(reinterpret_cast<const char*>(&num_hidden_layers), sizeof(num_hidden_layers));

    if (num_hidden_layers > 0) {
        std::vector<int> hidden_layers_sizes_nn;
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            hidden_layers_sizes_nn.push_back(layers[i].output_size);
        }
        outFile.write(reinterpret_cast<const char*>(hidden_layers_sizes_nn.data()),
                    sizeof(int) * hidden_layers_sizes_nn.size());
    }

    for (const auto& layer : layers) {

        std::vector<float> host_weights(layer.input_size * layer.output_size);
        queue.enqueueReadBuffer(layer.weights, CL_TRUE, 0, sizeof(float) * host_weights.size(), host_weights.data());
        outFile.write(reinterpret_cast<const char*>(host_weights.data()), sizeof(float) * host_weights.size());

        std::vector<float> host_biases(layer.output_size);
        queue.enqueueReadBuffer(layer.biases, CL_TRUE, 0, sizeof(float) * host_biases.size(), host_biases.data());
        outFile.write(reinterpret_cast<const char*>(host_biases.data()), sizeof(float) * host_biases.size());
    }

    outFile.close();
    std::cout << "Network saved to " << filename << std::endl;
    return 0;
}

NeuralNetwork NeuralNetwork::load(OpenCLSetup ocl_setup, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open file for loading: " << filename << std::endl;
        return NeuralNetwork();
    }

    float loaded_learning_rate;
    int loaded_batch_size;
    int loaded_input_size_nn;
    int loaded_output_size_nn;
    size_t loaded_num_hidden_layers;
    std::vector<int> loaded_hidden_layers_sizes_nn;

    inFile.read(reinterpret_cast<char*>(&loaded_learning_rate), sizeof(loaded_learning_rate));
    inFile.read(reinterpret_cast<char*>(&loaded_batch_size), sizeof(loaded_batch_size));
    inFile.read(reinterpret_cast<char*>(&loaded_input_size_nn), sizeof(loaded_input_size_nn));
    inFile.read(reinterpret_cast<char*>(&loaded_output_size_nn), sizeof(loaded_output_size_nn));
    inFile.read(reinterpret_cast<char*>(&loaded_num_hidden_layers), sizeof(loaded_num_hidden_layers));
    if (loaded_num_hidden_layers > 0) {
        loaded_hidden_layers_sizes_nn.resize(loaded_num_hidden_layers);
        inFile.read(reinterpret_cast<char*>(loaded_hidden_layers_sizes_nn.data()), sizeof(int) * loaded_num_hidden_layers);
    }

    NeuralNetwork loaded_net(ocl_setup, loaded_input_size_nn, loaded_hidden_layers_sizes_nn, loaded_output_size_nn, loaded_batch_size, loaded_learning_rate);

    for (auto& layer : loaded_net.layers) {
        std::vector<float> host_weights(layer.input_size * layer.output_size);
        inFile.read(reinterpret_cast<char*>(host_weights.data()), sizeof(float) * host_weights.size());
        layer.weights = cl::Buffer(ocl_setup.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * host_weights.size(), host_weights.data());

        std::vector<float> host_biases(layer.output_size);
        inFile.read(reinterpret_cast<char*>(host_biases.data()), sizeof(float) * host_biases.size());
        layer.biases = cl::Buffer(ocl_setup.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * host_biases.size(), host_biases.data());
    }

    inFile.close();
    std::cout << "Network loaded from " << filename << std::endl;
    return loaded_net;
}