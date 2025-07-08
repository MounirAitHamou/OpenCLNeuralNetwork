# OpenCLNeuralNetwork

**OpenCLNeuralNetwork** is a modular neural network framework written in modern C++ with OpenCL acceleration.  
This project was created as a personal learning project to understand neural network implementation, GPU programming, and OpenCL fundamentals from the ground up.

---

## 🚀 Features

- ⚙️ **Fully modular architecture**  
  Define any number of hidden layers, with full control over:
  - Hidden layer sizes
  - Activation functions per layer
  - Loss function type
  - Optimizer configuration
  - Batch size
  - Learning rate

- 🧠 **Current Components**
  - **Layer types:** Dense (more coming soon!)
    > Planned: Convolutional
  - **Activations:** Linear, Sigmoid, ReLU, Tanh
  - **Loss functions:** Mean Squared Error (MSE), Binary Cross Entropy (BCE)
  - **Optimizers:** Stochastic Gradient Descent (SGD), Adam, AdamW

- ⚡ **OpenCL-powered**
  - Matrix operations and training computations are offloaded to the GPU using OpenCL.
  - Enables accelerated training on compatible devices.

- 📦 **Batch training**
  - Supports training on mini-batches for better generalization and GPU parallelism.

---

## 📦 Project Status

This project is a **work in progress** and is being actively developed as a learning initiative.  
The goal is to implement core neural network functionality from scratch while gaining a deeper understanding of OpenCL and GPU computation.

---

## 🛣️ Roadmap

Planned features include:
- Input data preprocessing
- Convolutional layer support
- Saving/loading model weights
- More loss functions and activation types
- CLI/demo interfaces
- Automated differentiation support

---

## 📂 Documentation

- 🧰 [Installation Guide](./INSTALL.md)
- 🧪 [Usage Instructions](./USAGE.md)
- 📄 [License](./LICENSE)

---

## 🧑‍💻 Author

Built by Mounir Ait Hamou as a self-guided learning project.

---