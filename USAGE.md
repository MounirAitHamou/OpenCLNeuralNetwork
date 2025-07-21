# 🧪 Usage Guide – OpenCLNeuralNetwork

This guide walks through how to use and customize the `OpenCLNeuralNetwork` framework, including training, evaluating, and modifying your neural network configuration.

---

## 🏁 Running the XOR Example

The `makeXORModel()` function in `main.cpp` demonstrates how to train a small neural network to learn the XOR function using OpenCL-accelerated matrix operations.

## 🧪 Running the Regression Example

The `makeRegressionModel()` function in `main.cpp` shows how to train a neural network for regression tasks, such as predicting continuous values based on input features.

### ✅ To run the demo:

1. Make sure the project is built (see [INSTALL.md](./INSTALL.md)).
2. Run the compiled binary:
   ```bash
   OpenCLNeuralNetwork.exe
    ```


### Customizing the Neural Network

You can modify the neural network configuration by changing parameters in the `main()` function:
- **Batch Size**: Adjust the `batch_size` variable to change how many samples are processed in each training step.
- **Learning Rate**: Change the `learning_rate` variable to control how quickly the model learns.
- **Weight Decay Rate**: Set the `weight_decay_rate` to apply L2 regularization.
- **Epochs**: Modify the `epochs` variable to change how many times the model will iterate over the training data.
- **Loss Function**: Change the `loss_function_type` variable to switch between Mean Squared Error (MSE) and Binary Cross Entropy (BCE).
- **beta1, beta2, epsilon**: These parameters are used for the Adam and AdamW optimizers. Adjust them as needed.


🧩 Customizing the Optimizer
Currently, the framework supports Stochastic Gradient Descent (SGD), Adam and AdamW optimizers. You can configure the optimizer parameters as follows:

```cpp
const OptimizerConfig::SGDOptimizerParameters optimizer_parameters =
    OptimizerConfig::makeSGDParameters(
        learning_rate,
        weight_decay_rate
    );

const OptimizerConfig::AdamOptimizerParameters optimizer_parameters = OptimizerConfig::makeAdamParameters(
        learning_rate,
        weight_decay_rate,
        beta1,
        beta2,
        epsilon
    );

const OptimizerConfig::AdamWOptimizerParameters optimizer_parameters = OptimizerConfig::makeAdamWParameters(
        learning_rate,
        weight_decay_rate,
        beta1,
        beta2,
        epsilon
    );
```



🏗️ Customizing Layers and Network Architecture
The network architecture is defined by specifying layers using LayerArgs:

Adding Hidden Layers
```cpp
    std::vector<std::unique_ptr<LayerConfig::LayerArgs>> hidden_layers;
    hidden_layers.push_back(LayerConfig::makeDenseLayerArgs(
        Dimensions({4}), ActivationType::Tanh));  // Example hidden layer with 4 neurons and Tanh activation
 

    hidden_layers.push_back(LayerConfig::makeDenseLayerArgs(
        Dimensions({6}), ActivationType::ReLU));  // Second hidden layer example
```

Setting the Output Layer
```cpp
    std::unique_ptr<LayerConfig::LayerArgs> output_layer =
        LayerConfig::makeDenseLayerArgs(
            Dimensions({1}), ActivationType::Sigmoid);  // Output layer with 1 neuron and Sigmoid activation
```

Adding Layers to the Network after initialization
```cpp
    net.addDense(1, ActivationType::Sigmoid);  // Add output layer
```

💾 Saving and Loading a Network

You can save the trained network to a file and load it later using HDF5 format:

```cpp
    net.saveNetwork("network.h5");  // Save the network
```

```cpp
    NeuralNetwork loaded_net = NeuralNetwork::loadNetwork(ocl_setup, "network.h5");  // Load the network
```

📊 Data Processor

```cpp
    CSVNumericalProcessor csv_processor(ocl_setup, batch_size);
    csv_processor.loadData("data/Regression/test_regression1.csv", {"feature1", "feature2", "feature3"}, {"target"});
    unsigned int seed;
    seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    csv_processor.splitData(1.0f, 0.0f, seed);

    net.train(
        csv_processor,
        epochs
    );
```

Supported Activation Functions:
```cpp
ActivationType::Linear

ActivationType::Sigmoid

ActivationType::Tanh

ActivationType::ReLU
```

Supported Loss Functions:
```cpp
LossFunctionType::MeanSquaredError

LossFunctionType::BinaryCrossEntropy
```