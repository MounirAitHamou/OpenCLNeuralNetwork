#include <CL/opencl.hpp>
#include "OpenCLSetup.hpp"
#include "NeuralNetwork.hpp"


NeuralNetwork XORTrain(OpenCLSetup& ocl_setup) {
    int input_size = 2;
    const std::vector<int> hidden_layers_sizes = {4}; // One hidden layer with 4 neurons
    int output_size = 1;
    float learning_rate = 0.5f;
    int batch_size = 4;
    int epochs = 10000;

    if (input_size <= 0 || output_size <= 0 || learning_rate <= 0 || batch_size <= 0){
        std::cerr << "Error: Ensure hyperparameters are valid." << std::endl;
        throw std::runtime_error("Hyperparameters");
    }

    NeuralNetwork net(ocl_setup, input_size, hidden_layers_sizes, output_size, batch_size, learning_rate);
    net.initialize();

    // XOR "dataset" for training
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

    std::cout << "Training:\n";
    net.train(inputs, targets, epochs);

    return net;
}

int XORTest(OpenCLSetup& ocl_setup, NeuralNetwork& net) {
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

        int original_network_batch_size = net.batch_size;
        net.batch_size = 1;
        for (auto& layer : net.layers) {
            layer.batch_size = 1;
        }

        cl::Buffer output_buf = net.forward(test_input_buf);

        net.batch_size = original_network_batch_size;
        for (auto& layer : net.layers) {
            layer.batch_size = original_network_batch_size;
        }

        std::vector<float> prediction(1);
        ocl_setup.queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, sizeof(float), prediction.data());

        std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << prediction[0] <<
                  " (Target: " << targets[i][0] << ")" << "\n";
    }
    return 0;
}

int saveLoadExample(OpenCLSetup& ocl_setup, NeuralNetwork& net) {
    std::string filename = "xor_network.bin";

    if (net.save(filename) != 0) {
        std::cerr << "Error saving the network." << std::endl;
        return 1;
    }

    NeuralNetwork loaded_net = NeuralNetwork::load(ocl_setup, filename);
    if (!loaded_net.equalNetwork(net)) {
        std::cerr << "Error: Loaded network does not match the original." << std::endl;
        return 1;
    } else {
        std::cout << "Network loaded successfully and matches the original." << std::endl;
    }

    std::cout << "\nTesting original network:\n";
    XORTest(ocl_setup, net);

    std::cout << "\nTesting loaded network:\n";
    XORTest(ocl_setup, loaded_net);

    return 0;
}


int loadNetworkFromFile() {
    std::string filename = "xor_network.bin";
    OpenCLSetup ocl_setup = OpenCLSetup::createOpenCLSetup();
    NeuralNetwork net;
    try {
        net = NeuralNetwork::load(ocl_setup, filename);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error loading network from file: " << e.what() << std::endl;
        return 1;
    }
    XORTest(ocl_setup, net);
    return 0;
}


int main() {
    OpenCLSetup ocl_setup;
    try {
        ocl_setup = OpenCLSetup::createOpenCLSetup();
    } catch (const std::runtime_error& e) {
        std::cerr << "OpenCL setup failed: " << e.what() << std::endl;
        return 1;
    }

    
    // Test examples
    NeuralNetwork net = XORTrain(ocl_setup);

    if (saveLoadExample(ocl_setup, net) != 0) {
        std::cerr << "Error in save/load example." << std::endl;
        return 1;
    }

    //loadNetworkFromFile();


    std::cout << "Press Enter to exit...\n";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    std::cout << "Exiting...\n";
    return 0;
}