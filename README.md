# OpenCLNeuralNetwork

**OpenCLNeuralNetwork** is a modular neural network framework written in modern C++ with OpenCL acceleration.  
This project was created as a personal learning project to understand neural network implementation, GPU programming, and OpenCL fundamentals from the ground up.

---

## 🚀 Features

- ⚙️ **Fully modular architecture**  
  Define any number of hidden layers, with full control over:
  - Layer types
  - Layer sizes
  - Loss function type
  - Optimizer type and configuration
  - Batch size

- 🧠 **Current Components**
  - **Layer types:** Dense, Convolutional, Softmax, Sigmoid, ReLU, Tanh, LeakyReLU
  - **Loss functions:** Mean Squared Error (MSE), Binary Cross Entropy (BCE)
  - **Optimizers:** Stochastic Gradient Descent (SGD), Adam, AdamW

- ⚡ **OpenCL-powered**
  - Matrix operations and training computations are offloaded to the GPU using OpenCL.
  - Enables accelerated training on compatible devices.

- 🧮 **CLBlast integration**
  - Utilizes CLBlast for optimized matrix multiplications and other linear algebra operations.

- 📦 **Batch training**
  - Supports training on mini-batches for better generalization and GPU parallelism.

- 💾 **Model saving/loading**
  - Save and load model configurations and weights in HDF5 format for easy persistence.

- 📊 **Data processing**
  - Built-in CSV data loader for loading and preprocessing datasets.
    > Planned: Image data processor
  - Supports splitting data into training, validation, and test sets.

---

## 📦 Project Status

This project is a **work in progress** and is being actively developed as a learning initiative.  
The goal is to implement core neural network functionality from scratch while gaining a deeper understanding of OpenCL and GPU computation.

---

## 🛣️ Roadmap

Planned features include:
- 📈 Additional layer types (e.g., Dropout, Batch Normalization)
---

## 📂 Documentation

- 🧰 [Installation Guide](./INSTALL.md)
- 🧪 [Usage Instructions](./USAGE.md)
- 📄 [License](./LICENSE)

---

## 🧑‍💻 Author

Built by Mounir Ait Hamou as a self-guided learning project.

---

## 🤝 Collaboration

I'm open to collaborations! The best way to get in touch is directly through this repository:  

- Open an [Issue](https://github.com/MounirAitHamou/OpenCLNeuralNetwork/issues)  
- Start or reply in [Discussions](https://github.com/MounirAitHamou/OpenCLNeuralNetwork/discussions)  

All communication can happen through GitHub—no email needed. I'm happy to review contributions, answer questions, or discuss ideas!

---