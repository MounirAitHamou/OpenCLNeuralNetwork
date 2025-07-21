#include "NeuralNetwork/NeuralNetwork.hpp"


void mainPrintCLBuffer(cl::CommandQueue queue, const cl::Buffer& buffer, size_t size, const std::string& label = "Buffer") {
    std::vector<float> host_data(size);
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(float) * size, host_data.data());
    std::cout << label << " Buffer Data: ";
    for (const auto& value : host_data) {
        std::cout << value << " ";
    }
    std::cout << "\n";
}

int XORTest(OpenCLSetup ocl_setup, NeuralNetwork& net){
    size_t batch_size = 1;
    CSVNumericalProcessor csv_processor(ocl_setup, batch_size);
    csv_processor.loadData("data/XOR/xor_data.csv", {"bit1", "bit2"}, {"outputbit"});
    unsigned int seed;
    seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    csv_processor.splitData(1.0f, 0.0f, seed);
    const Dimensions initial_input_dims = Dimensions({csv_processor.getInputSize()});
    const size_t initial_input_dims_size = initial_input_dims.getTotalElements();
    const size_t output_dims = csv_processor.getTargetSize();


    std::cout << "\nTesting:\n";
    for (const Batch& batch : csv_processor){
        cl::Buffer test_input_buf = batch.inputs;
        cl::Buffer targets = batch.targets;

        size_t original_network_batch_size = net.batch_size;
        net.batch_size = 1;

        cl::Buffer output_buf = net.forward(test_input_buf, 1);

        net.batch_size = original_network_batch_size;

        std::vector<float> prediction(output_dims);
        ocl_setup.queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, sizeof(float) * output_dims, prediction.data());

        std::vector<float> target(output_dims);
        ocl_setup.queue.enqueueReadBuffer(targets, CL_TRUE, 0, sizeof(float) * output_dims, target.data());

        std::vector<float> input_data(initial_input_dims_size);
        ocl_setup.queue.enqueueReadBuffer(test_input_buf, CL_TRUE, 0, sizeof(float) * initial_input_dims_size, input_data.data());

        std::cout << "Input: (" << input_data[0] << ", " << input_data[1] << ")"
          << " | Predicted: " << prediction[0]
          << " | Target: " << target[0] << std::endl;
    }
    return 0;
}

int RegressionTest(OpenCLSetup ocl_setup, NeuralNetwork& net){
    size_t batch_size = 1;
    CSVNumericalProcessor csv_processor(ocl_setup, batch_size);
    csv_processor.loadData("data/Regression/test_regression1.csv", {"feature1", "feature2", "feature3"}, {"target"});
    unsigned int seed;
    seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    csv_processor.splitData(1.0f, 0.0f, seed);
    const Dimensions initial_input_dims = Dimensions({csv_processor.getInputSize()});
    const size_t initial_input_dims_size = initial_input_dims.getTotalElements();
    const size_t output_dims = csv_processor.getTargetSize();

    std::cout << "\nTesting:\n";
    for (const Batch& batch : csv_processor){
        cl::Buffer test_input_buf = batch.inputs;
        cl::Buffer targets = batch.targets;

        size_t original_network_batch_size = net.batch_size;
        net.batch_size = 1;

        cl::Buffer output_buf = net.forward(test_input_buf, 1);

        net.batch_size = original_network_batch_size;

        std::vector<float> prediction(output_dims);
        ocl_setup.queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, sizeof(float) * output_dims, prediction.data());

        std::vector<float> target(output_dims);
        ocl_setup.queue.enqueueReadBuffer(targets, CL_TRUE, 0, sizeof(float) * output_dims, target.data());

        std::vector<float> input_data(initial_input_dims_size);
        ocl_setup.queue.enqueueReadBuffer(test_input_buf, CL_TRUE, 0, sizeof(float) * initial_input_dims_size, input_data.data());

        std::cout << "Input: (" << input_data[0] << ", " << input_data[1] << ", " << input_data[2] << ")"
          << " | Predicted: " << prediction[0]
          << " | Target: " << target[0] << std::endl;
    }
    return 0;
}

int makeXORModel(OpenCLSetup ocl_setup, const std::string filename){
    size_t batch_size = 3;
    float learning_rate = 0.01f;
    float weight_decay_rate = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int epochs = 10000;


    
    CSVNumericalProcessor csv_processor(ocl_setup, batch_size);
    csv_processor.loadData("data/XOR/xor_data.csv", {"bit1", "bit2"}, {"outputbit"});
    unsigned int seed;
    seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    csv_processor.splitData(1.0f, 0.0f, seed);
    const Dimensions initial_input_dims = Dimensions({csv_processor.getInputSize()});
    const size_t initial_input_dims_size = initial_input_dims.getTotalElements();
    const size_t output_dims = csv_processor.getTargetSize();

    if (!std::filesystem::exists(filename)){
        LossFunctionType loss_function_type = LossFunctionType::BinaryCrossEntropy;
        
        std::vector<std::unique_ptr<LayerConfig::LayerArgs>> hidden_layers;
        
        /*
        This is an alternative way of defining the network,
        hidden_layers.push_back(LayerConfig::makeDenseLayerArgs(
            Dimensions({2}), ActivationType::Linear));

        std::unique_ptr<LayerConfig::LayerArgs> output_layer = LayerConfig::makeDenseLayerArgs(
            Dimensions({4}), ActivationType::Tanh);

        NeuralNetwork net(ocl_setup, NetworkConfig::createNetworkArgs( 
            initial_input_dims,
            std::move(hidden_layers),
            std::move(output_layer),   
            optimizer_parameters,
            batch_size,
            loss_function_type
        ));
        */ 

        auto optimizer_parameters = OptimizerConfig::makeAdamWParameters(
            learning_rate,
            weight_decay_rate,
            beta1,
            beta2,
            epsilon
        );

        NeuralNetwork net(ocl_setup, NetworkConfig::createNetworkArgs( 
            initial_input_dims,    
            optimizer_parameters,
            batch_size,
            loss_function_type
        ));

        net.addDense(initial_input_dims_size, ActivationType::Linear)
           .addDense(4, ActivationType::Tanh)
           .addDense(4, ActivationType::ReLU)
           .addDense(output_dims, ActivationType::Sigmoid); // A neural network with 2 inputs, a linear hidden layer with 2 neurons, a Tanh layer with 4 neurons, a ReLU layer with 4 neurons, and a Sigmoid output layer.

        net.train(
            csv_processor,
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
            csv_processor,
            epochs
        );

        std::cout << "\nTesting after retraining:\n";
        XORTest(ocl_setup, loaded_net);

        std::cout << "\nCheckpointing again...\n";
        loaded_net.saveNetwork(filename);
    }
}

int makeRegressionModel(OpenCLSetup ocl_setup, const std::string filename) {
    size_t batch_size = 3;
    float learning_rate = 0.01f;
    float weight_decay_rate = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int epochs = 10000;

    CSVNumericalProcessor csv_processor(ocl_setup, batch_size);
    csv_processor.loadData("data/Regression/test_regression1.csv", {"feature1", "feature2", "feature3"}, {"target"});
    unsigned int seed;
    seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    csv_processor.splitData(1.0f, 0.0f, seed);
    const Dimensions initial_input_dims = Dimensions({csv_processor.getInputSize()});
    const size_t initial_input_dims_size = initial_input_dims.getTotalElements();
    
    const size_t output_dims = csv_processor.getTargetSize();


    if (!std::filesystem::exists(filename)){
        LossFunctionType loss_function_type = LossFunctionType::MeanSquaredError;
        
        std::vector<std::unique_ptr<LayerConfig::LayerArgs>> hidden_layers;

        auto optimizer_parameters = OptimizerConfig::makeAdamWParameters(
            learning_rate,
            weight_decay_rate,
            beta1,
            beta2,
            epsilon
        );

        NeuralNetwork net(ocl_setup, NetworkConfig::createNetworkArgs( 
            initial_input_dims,    
            optimizer_parameters,
            batch_size,
            loss_function_type
        ));

        net.addDense(initial_input_dims_size, ActivationType::ReLU)
           .addDense(8, ActivationType::Tanh)
           .addDense(4, ActivationType::ReLU)
           .addDense(output_dims, ActivationType::Linear);

        std::cout << "Training network...\n";

        net.train(
            csv_processor,
            epochs
        );
        RegressionTest(ocl_setup, net);
        std::cout << "\nSaving network to file\n";
        net.saveNetwork(filename);
    }

    NeuralNetwork loaded_net = NeuralNetwork::loadNetwork(ocl_setup, filename);

    // Checkpoint training loop
    while (true){
        std::cout << "\nTesting loaded network:\n";
        RegressionTest(ocl_setup, loaded_net);

        loaded_net.train(
            csv_processor,
            epochs
        );

        std::cout << "\nTesting after retraining:\n";
        RegressionTest(ocl_setup, loaded_net);

        std::cout << "\nCheckpointing again...\n";
        loaded_net.saveNetwork(filename);
    }
    
}


int main() {
    const OpenCLSetup ocl_setup = OpenCLSetup::createOpenCLSetup("kernels");

    //makeXORModel(ocl_setup, "xor_network.h5");
    makeRegressionModel(ocl_setup, "regression_network.h5");
    return 0;
}