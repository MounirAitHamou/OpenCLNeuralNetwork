This project implements a simple neural network that leverages OpenCL for accelerated computations, specifically designed for batched training. It demonstrates the core concepts of a feedforward neural network, including initialization, forward propagation, backpropagation, and weight updates, all offloaded to a GPU or CPU using OpenCL. The project includes an example of training the network to solve the XOR problem and functionalities to save and load the trained network.

Features
OpenCL Acceleration: Utilizes OpenCL to parallelize neural network computations (forward pass, backpropagation, weight updates) on compatible devices (GPUs or CPUs).

Batched Training: Implements training with mini-batches for more efficient learning.

Configurable Network Architecture: Supports defining the number of hidden layers and neurons per hidden layer.

XOR Problem Example: Includes a complete example of training and testing the network on the XOR logical operation.

Network Persistence: Ability to save the trained neural network's weights and biases to a file and load them back.

Sigmoid Activation: Uses the sigmoid activation function for neurons.

main.cpp: Contains the main application logic, including OpenCL setup, XOR training and testing examples, and network save/load demonstrations.

NeuralNetwork.hpp/NeuralNetwork.cpp: Defines the NeuralNetwork class, encapsulating the entire neural network structure and core functionalities like forward, backprop, train, save, and load.

LayerGPU.hpp/LayerGPU.cpp: Defines the LayerGPU struct, representing a single layer within the neural network, responsible for managing its weights, biases, and OpenCL buffers. It includes methods for initialization (setRandomParams) and running the forward pass (runForward).

OpenCLSetup.hpp/OpenCLSetup.cpp: Handles the initial setup of the OpenCL environment, including platform and device selection, context creation, command queue creation, and OpenCL program compilation.

kernels/kernels.cl: Contains the OpenCL kernel code for parallel computations, including:

activate: Sigmoid activation function.

activate_derivative: Derivative of the sigmoid activation function.

layer_forward_batch: Kernel for performing the forward pass for a layer on a batch of inputs.

compute_output_delta_batch: Kernel for calculating the error deltas for the output layer.

backpropagate_delta_batch: Kernel for backpropagating error deltas through hidden layers.

update_weights_batch: Kernel for updating weights and biases based on calculated deltas and learning rate.

How it Works
The neural network implementation follows a standard feedforward architecture with backpropagation for training.

OpenCL Initialization: The setupOpenCL function in main.cpp discovers available OpenCL platforms and devices, prompts the user to select a device, creates an OpenCL context and command queue, and compiles the OpenCL kernels from kernels.cl.

Neural Network Construction: The NeuralNetwork class is initialized with input size, hidden layer sizes, output size, batch size, and learning rate. It then creates a series of LayerGPU objects, each representing a layer in the network.

Initialization: The initialize method of NeuralNetwork calls setRandomParams for each LayerGPU to initialize weights and biases with random values.

Forward Pass (forward):

Input data (a batch) is copied to an OpenCL buffer.

For each layer, the runForward method is called.

runForward executes the layer_forward_batch kernel on the OpenCL device.

The layer_forward_batch kernel calculates the weighted sum of inputs plus bias for each neuron in parallel and applies the sigmoid activation function to produce the layer's outputs.

The output of one layer becomes the input for the next.

Training (train):

The train method iterates through a specified number of epochs.

In each epoch, the training data is shuffled and processed in mini-batches.

For each batch:

Inputs and targets are copied to OpenCL buffers.

A forward pass is performed.

The loss is calculated.

Backpropagation (backprop):

The compute_output_delta_batch kernel calculates the error delta for the output layer.

The backpropagate_delta_batch kernel propagates these deltas backward through the hidden layers.

Finally, the update_weights_batch kernel adjusts the weights and biases of each layer based on the calculated deltas and the learning rate.

Save/Load: The save method serializes the network's hyperparameters and the weights and biases of all layers into a binary file. The load method reconstructs the network from such a file.

Setup instructions for OpenCL Neural Network

1. Install the required dependencies:
   - Install C++ compiler (I used Visual Studio 2022 build tools).
      - If using MSVC, make sure to be compiling with the x64 architecture.

   - Install CMake (https://cmake.org/download/).

   - Build OpenCL SDK in Visual Studio 2022 build tools:
     - For Intel, I had to install Vulkan:
        - https://vulkan.lunarg.com/sdk/home
        - Follow the instructions to install the Vulkan SDK.

     - Pull the OpenCL SDK from the repository https://github.com/KhronosGroup/OpenCL-SDK.git
     - Navigate to the directory that you pulled the OpenCL SDK to. (For example, I pulled it to C:/GitHub/OpenCL-SDK, so I navigated to C:/GitHub)
     - cmake -A x64 -D CMAKE_INSTALL_PREFIX=./OpenCL-SDK/install -B ./OpenCL-SDK/build -S ./OpenCL-SDK
     - cmake --build OpenCL-SDK/build --config Release --target install -- /m /v:minimal

     - Navigate to the root directory (Learning/MachineLearning/OpenCLNeuralNetwork), in CMAKELists.txt, define your OpenCL SDK install path.
        - Mine was "C:/GitHub/OpenCL-SDK/install"

2. Run rebuild.bat in the OpenCLNeuralNetwork directory to build the project.
3. Run OpenCLNeuralNetwork.exe to test the setup.

The program will:

List available OpenCL devices.

Prompt you to select a device by index.

Train a neural network on the XOR dataset.

Test the trained network on XOR inputs.

Demonstrate saving the network to xor_network.bin and loading it back, then testing the loaded network for verification.

Wait for user input before exiting.