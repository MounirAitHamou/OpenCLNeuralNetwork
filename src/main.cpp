#include "NeuralNetwork/NeuralNetwork.hpp"

int XORTest(OpenCLSetup ocl_setup, NeuralNetwork& net) {
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    std::vector<std::vector<float>> targets = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    std::cout << "\nTesting:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        cl::Buffer test_input_buf(ocl_setup.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * inputs[i].size(), (void*)inputs[i].data());

        size_t original_network_batch_size = net.batch_size;
        net.batch_size = 1;

        cl::Buffer output_buf = net.forward(test_input_buf, 1);

        net.batch_size = original_network_batch_size;

        

        std::vector<float> prediction(1);
        ocl_setup.queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, sizeof(float), prediction.data());

        std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << prediction[0] <<
                  " (Target: " << targets[i][0] << ")" << "\n";
    }
    return 0;
}

int makeXORModel(OpenCLSetup ocl_setup, const std::string filename){
    size_t batch_size = 4;
    float learning_rate = 0.01f;
    float weight_decay_rate = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int epochs = 10000;
    if (!std::filesystem::exists(filename)){
        LossFunctionType loss_function_type = LossFunctionType::BinaryCrossEntropy;

        const Dimensions initial_input_dims = Dimensions({2});

        std::vector<std::unique_ptr<LayerConfig::LayerArgs>> hidden_layers;

        hidden_layers.push_back(LayerConfig::makeDenseLayerArgs(
            Dimensions({2}), ActivationType::Linear));

        hidden_layers.push_back(LayerConfig::makeDenseLayerArgs(
            Dimensions({4}), ActivationType::Tanh));

        hidden_layers.push_back(LayerConfig::makeDenseLayerArgs(
            Dimensions({4}), ActivationType::ReLU));

        std::unique_ptr<LayerConfig::LayerArgs> output_layer = LayerConfig::makeDenseLayerArgs(
            Dimensions({1}), ActivationType::Sigmoid);
        
        const OptimizerConfig::AdamWOptimizerParameters optimizer_parameters = OptimizerConfig::makeAdamWParameters(
            learning_rate,
            weight_decay_rate,
            beta1,
            beta2,
            epsilon
        );

        NeuralNetwork net(ocl_setup, NetworkConfig::createNetworkArgs( 
            initial_input_dims,
            std::move(hidden_layers), 
            std::move(output_layer),       
            optimizer_parameters,
            batch_size,
            loss_function_type
        ));

        net.train(
            {
                {0.0f, 0.0f},
                {0.0f, 1.0f},
                {1.0f, 0.0f},
                {1.0f, 1.0f}
            },
            {
                {0.0f},
                {1.0f},
                {1.0f},
                {0.0f}
            },
            epochs
        );

        XORTest(ocl_setup, net);
        std::cout << "\nSaving network to file\n";
        net.saveNetwork(filename);
    }

    NeuralNetwork loaded_net = NeuralNetwork::loadNetwork(ocl_setup, filename);

    // Checkpoint training loop
    while (true){
        std::cout << "\nTesting loaded network:\n";
        XORTest(ocl_setup, loaded_net);

        loaded_net.train(
            {
                {0.0f, 0.0f},
                {0.0f, 1.0f},
                {1.0f, 0.0f},
                {1.0f, 1.0f}
            },
            {
                {0.0f},
                {1.0f},
                {1.0f},
                {0.0f}
            },
            epochs
        );

        std::cout << "\nTesting after retraining:\n";
        XORTest(ocl_setup, loaded_net);

        std::cout << "\nCheckpointing again...\n";
        loaded_net.saveNetwork(filename);
    }
}

int main() {
    const OpenCLSetup ocl_setup = OpenCLSetup::createOpenCLSetup("kernels");

    makeXORModel(ocl_setup, "xor_network.h5");
    
    return 0;
}