# OpenCLNeuralNetwork

## Overview

This project is a **from-scratch GPU-accelerated neural network framework** written in modern C++20 using OpenCL. It implements both forward and backward propagation entirely on the GPU without relying on external machine learning libraries such as TensorFlow, PyTorch, or Eigen.

The framework is designed to expose low-level control over computation, memory, and execution while still supporting a modular deep learning architecture.

It supports:
- Dense (fully connected) layers
- Convolutional layers
- Activation layers (ReLU, LeakyReLU, Sigmoid, Tanh, Softmax)
- Multiple loss functions (MSE, Binary Cross Entropy, Categorical Cross Entropy, Softmax Cross Entropy)
- Optimizers (SGD, Adam, AdamW)
- Dataset loaders (CIFAR-10, CSV regression, XOR dataset)
- Model serialization via HDF5
- Runtime visualization of layer tensors and activations

---

## Core Design Philosophy

### 1. Explicit GPU Control
All computation is explicitly managed using:
- OpenCL buffers (`cl::Buffer`)
- Command queues
- Event-based synchronization

There is no abstraction hiding memory movement or kernel execution.

---

### 2. Modular Layer Graph
Neural networks are represented as a sequential graph of polymorphic `Layer` objects.

Each layer:
- Owns its own GPU buffers (`outputs`, `deltas`)
- Implements forward and backward propagation
- Exposes a unified interface for execution

Base abstraction:

```cpp
class Layer {
    virtual cl::Event runForward(...)
    virtual cl::Event backpropDeltas(...)
};
```

## Trainable Layers Extension

Trainable layers extend the base `Layer` abstraction with:

- Weight tensors (GPU buffers)
- Bias tensors (GPU buffers)
- Gradient computation logic (forward + backward pass derivatives)
- Optimizer hooks for parameter updates (SGD, Adam, AdamW integration)

These layers are responsible for all learnable parameters and their updates during backpropagation.

---

## Multi-Queue OpenCL Execution Model

The system uses three OpenCL command queues to structure execution:

- **Forward/Backprop Queue** → main inference and gradient propagation
- **Delta-to-Gradient Queue** → weight and bias gradient computation
- **Concurrent Queue** → loss computation and auxiliary operations

This separation enables partial overlap of compute workloads and reduces synchronization bottlenecks across training stages. Since CLBlast has no wait-event support, multiple queues + barriers are used to ensure correct ordering while maximizing GPU utilization.

---

## System Architecture

### Neural Network Core

The `LocalNeuralNetwork` class manages:

- Construction of layer graphs from configuration
- Forward propagation through chained OpenCL kernels
- Backpropagation via explicit delta propagation between layers
- Gradient computation for trainable parameters
- Optimizer updates executed on the GPU
- Full training loop execution over dataset loaders
- Model serialization and deserialization (HDF5-based)

---

## Layer System

### Layer Hierarchy

```text
Layer (abstract)
├── ActivationLayer
│   ├── ReLU
│   ├── LeakyReLU
│   ├── Sigmoid
│   ├── Tanh
│   └── Softmax
└── TrainableLayer
    ├── DenseLayer
    └── ConvolutionalLayer
```

## Layer Responsibilities

Each layer is responsible for:

- Storing output tensor dimensions
- Allocating GPU buffers:
  - `outputs`
  - `deltas`
- Implementing forward and backward OpenCL kernels
- Supporting serialization to disk (HDF5)
- Supporting runtime visualization of internal state

---

## Convolution Implementation

Convolution is implemented with CLBlast for the forward pass, while backpropagation of deltas and weight gradient computation are implemented with custom OpenCL kernels that manually handle convolution indexing, stride, and padding logic.

## Dense Implementation
Dense layer utilizes CLBlast for matrix multiplication in the forward pass, backpropagation of deltas, and weight gradient computation.

### Key kernels:

#### Backpropagation of deltas
- Computes gradients with respect to previous layers
- Manually handles stride, padding, and channel mapping
- Uses vectorized operations (`float4`) for performance optimization

#### Weight gradient computation
- Iterates over batch, spatial dimensions, and filter positions
- Accumulates contributions from input × delta products

#### Bias gradient computation
- Reduces deltas across spatial dimensions and batch size

---

## Optimization System

Supported optimizers:

### SGD
- Standard gradient descent update
- Optional L2 weight decay

### Adam
- First and second moment tracking
- Bias correction
- Numerical stability via epsilon term

### AdamW
- Decoupled weight decay formulation

All optimizers are implemented as OpenCL kernels that directly update GPU-resident parameter buffers.

---

## Memory Model

Each layer owns:

- `outputs` buffer
- `deltas` buffer

Trainable layers additionally own:

- Weight buffers
- Bias buffers
- Optimizer state buffers (e.g., `m`, `v` for Adam)

### Key properties:

- No implicit memory copying between layers
- Batch-size aware buffer allocation
- Dynamic resizing when batch size changes

---

## Execution Model

### Forward Pass
- Input buffer enters the first layer
- Each layer executes `runForward`
- Output is passed as input to the next layer
- Final layer produces predictions

### Backward Pass
- Loss gradient computed at output layer
- Deltas propagate backward through the network
- Trainable layers compute parameter gradients
- Optimizer updates are applied on GPU

---

## Data Pipeline

Supported dataset formats:

- CIFAR-10 binary dataset
- CSV regression datasets
- XOR synthetic dataset

Data is wrapped into `Batch` objects containing:

- Input buffers
- Target buffers
- Metadata (dimensions, batch size)

---

## Persistence (HDF5)

The framework supports full model serialization:

- Network architecture
- Layer parameters
- Optimizer state
- RNG state

### This enables:

- Full training resumption
- Deterministic reproducibility

---

## Visualization System

Each layer can generate runtime visualizations of:

- Outputs
- Deltas
- Metadata (batch size, dimensions, layer type)

### Supported features:

- Single-sample extraction
- Batch-level inspection
- GPU buffer-backed tensor visualization

---

## OpenCL Kernels

The framework includes custom kernels for:

### Activation functions
- All supported activations (ReLU, LeakyReLU, Sigmoid, Tanh, Softmax) have GPU implementations for both forward and backward passes.

### Convolution
- Forward computation (layer-specific implementations)
- Backpropagation of deltas
- Weight gradient computation
- Bias gradient computation

### Optimizers
- SGD update
- Adam update
- AdamW update

All kernels operate directly on GPU memory without CPU staging.

---

## Key Technical Characteristics

- Fully custom OpenCL compute backend
- Manual memory management (no ML frameworks)
- Explicit multi-queue execution model
- GPU-resident training loop
- Modular layer abstraction system
- HDF5-based persistence and serialization
- Vectorized convolution kernels
- Deterministic RNG initialization support

---

## Engineering Complexity Highlights

This project implements:

- GPU-based backpropagation from scratch
- Manual convolution indexing with stride/padding logic
- Multi-stream OpenCL execution pipeline
- GPU-resident optimizer state management
- Asynchronous loss computation
- Dynamic batch resizing at runtime
- Full model serialization and reconstruction system

---

## Summary

This is a low-level GPU neural network framework that exposes the entire machine learning pipeline—from tensor operations to optimization updates—directly at the OpenCL kernel level, with full explicit control over memory, execution, and synchronization.