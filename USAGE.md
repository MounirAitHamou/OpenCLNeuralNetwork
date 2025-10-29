# 🧪 Usage Guide – OpenCLNeuralNetwork

This guide walks through how to use and customize the `OpenCLNeuralNetwork` framework, including training, evaluating, and modifying your neural network configuration.

---

## 🏁 Running the XOR Example

The `makeXORModel()` function in `main.cpp` demonstrates how to train a small neural network to learn the XOR function using OpenCL-accelerated matrix operations.

### ✅ To run the demo:

1. Make sure the project is built (see [INSTALL.md](./INSTALL.md)).
2. Run the compiled binary:
   ```bash
   OpenCLNeuralNetwork.exe
    ```


### Customizing the Neural Network

You can modify the neural network configuration by changing parameters in the `main()` function:
- **Batch Size**: Adjust the `batchSize` variable to change how many samples are processed in each training step.
- **Learning Rate**: Change the `learningRate` variable to control how quickly the model learns.
- **Weight Decay Rate**: Set the `weightDecayRate` to apply L2 regularization.
- **Epochs**: Modify the `epochs` variable to change how many times the model will iterate over the training data.
- **Loss Function**: Change the `lossFunctionType` variable to switch between Mean Squared Error (MSE) and Binary Cross Entropy (BCE).
- **beta1, beta2, epsilon**: These parameters are used for the Adam and AdamW optimizers. Adjust them as needed.


🧩 Customizing the Optimizer
Currently, the framework supports Stochastic Gradient Descent (SGD), Adam and AdamW optimizers. You can configure the optimizer parameters as follows:

```cpp
auto optimizerArgs = Utils::makeAdamWArgs(
            learningRate,
            weightDecayRate,
            beta1,
            beta2,
            epsilon
        );
```



🏗️ Customizing Layers and Network Architecture
The network architecture is defined by specifying layers using LayerArgs:

Adding Layers:
```cpp
    std::vector<std::unique_ptr<LayerConfig::LayerArgs>> layers;
    layers.push_back(LayerConfig::makeDenseLayerArgs(
        Dimensions({4}));  // First hidden layer example
 

    layers.push_back(LayerConfig::makeDenseLayerArgs(
        Dimensions({6}), ActivationType::ReLU));  // Second hidden layer example
```


Adding Layers to the Network after initialization
```cpp
    net.addDense(3);
    net.addReLU();
    net.addConvolutional(Utils::FilterDimensions(1, 1, 3, 8), Utils::StrideDimensions(1, 1), Utils::PaddingType::Same);
    net.addSoftmax();
```

💾 Saving and Loading a Network

You can save the trained network to a file and load it later using HDF5 format:

```cpp
    net.saveNetwork("network.h5");  // Save the network
```

```cpp
    NeuralNetwork loadedNet = NeuralNetwork::loadNetwork(oclResources.getSharedResources(), "network.h5");
```

📊 Data Processor

```cpp
    CSVNumericalLoader csvLoader(oclResources.getSharedResources(), batchSize);
    csvLoader.loadData("data/XOR/xor_data.csv", {"bit1", "bit2"}, {"outputbit"});
    size_t seed;
    seed = static_cast<size_t>(std::chrono::system_clock::now().time_since_epoch().count());
    csvLoader.splitData(1.0f, 0.0f, seed);

    net.train(
        csvLoader,
        epochs,
        lossReporting
    );
```

Supported Loss Functions:
```cpp
LossFunctionType::MeanSquaredError

LossFunctionType::BinaryCrossEntropy
```