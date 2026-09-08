# CLNN

CLNN is a C++20, OpenCL-accelerated neural-network library built around a dynamic computation graph. Tensor operations record their dependencies, and `backward()` dispatches reverse-mode automatic differentiation kernels to the selected GPU. Modules and optimizers consume that graph instead of owning custom backpropagation pipelines.

This is the 2.0 rewrite of the original OpenCLNeuralNetwork project. GPU tensors own OpenCL buffers, while a CPU backend provides a deterministic correctness oracle and a portable fallback for tests.

## Highlights

- Dynamic, define-by-run computation graphs with automatic topological traversal
- Runtime-loaded OpenCL 1.2 backend with cached kernels, pooled GPU buffers, and event profiling
- Asynchronous, in-order OpenCL kernel submission with explicit device synchronization
- GPU-resident values, gradient accumulation, optimizer moments, and update kernels
- Gradient accumulation through branches and NumPy-style broadcasting
- In-place version checks, graph lifetime checks, and `NoGradGuard`
- Elementwise arithmetic, tiled matrix multiplication, 2D convolution and pooling, axis reductions, reshape/transpose, softmax, and common nonlinearities
- Stable MSE, binary cross-entropy, BCE-with-logits, and cross-entropy losses
- `Linear`, `Conv2d`, pooling, `BatchNorm`, `LayerNorm`, `Dropout`, residual blocks, global average pooling, activations, and owning `Sequential` modules
- SGD with momentum, Adam, and decoupled AdamW, with parameter groups and learning-rate schedulers
- Named, shape-checked, dependency-free model state files
- Full sequential architecture checkpoints and resumable optimizer moment state
- CSV numerical and CIFAR-10 loaders, deterministic splits, shuffling, mini-batches, and asynchronous prefetch
- Tensor health/statistics inspection on CPU or GPU
- Typed Python bindings with NumPy interoperability and native GPU execution
- Saliency, Input x Gradient, Integrated Gradients, SmoothGrad, and removable module hooks
- Deterministic unit, gradient, serialization, optimizer, and end-to-end training tests
- Built-in CPU/OpenCL microbenchmarks for matrix, convolution/pooling, and reduction kernels
- Installable CMake target: `CLNN::clnn`

## Quick start

```cpp
#include <clnn/clnn.hpp>

#include <memory>
#include <random>

std::mt19937 rng(42);
auto gpu = clnn::Device::opencl();
clnn::nn::Sequential model;
model.add(std::make_unique<clnn::nn::Linear>(2, 8, rng, true, gpu))
     .add(std::make_unique<clnn::nn::Tanh>())
     .add(std::make_unique<clnn::nn::Linear>(8, 1, rng, true, gpu));

clnn::Tensor x({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2}, false, {}, gpu);
clnn::Tensor y({0, 1, 1, 0}, {4, 1}, false, {}, gpu);
clnn::optim::Adam optimizer(model.parameters(), 0.03F);

optimizer.zero_grad();
auto loss = clnn::binary_cross_entropy_with_logits(model(x), y);
loss.backward();
optimizer.step();
```

## Build and test

CLNN loads the operating system's OpenCL loader dynamically, so compiling does not require OpenCL headers or an SDK. Running GPU code requires a vendor OpenCL driver. Tests use the vendored GoogleTest checkout.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Run the XOR example with `out/build/debug/clnn_examples` (or `clnn_examples.exe` on Windows).
The equivalent Python example, including all four explainability methods, is available at
`python/examples/clnn_xor.py`.
Run `clnn_examples --cifar` to train the convolutional CIFAR-10 example from
`data/CIFAR-10/data_batch_1.bin`; it resumes `cifar10.clnn` and its AdamW state when present.
Run `out/build/release/clnn_benchmarks` to compare synchronized CPU and OpenCL operation batches
and print per-kernel event timings. `clnn_benchmarks --check` applies the conservative regression
limits used by CTest and CI.

## Python

Build and install the Python package from the repository root:

```sh
python -m pip install .
```

The Python tensors use the same C++ autograd graph and OpenCL kernels as the native API:

```python
import numpy as np
import clnn
from clnn import nn, optim

device = clnn.Device.opencl() if clnn.opencl_available() else clnn.Device.cpu()
model = nn.Sequential()
model.add(nn.Linear(2, 8, seed=42, device=device))
model.add(nn.Tanh())
model.add(nn.Linear(8, 1, seed=43, device=device))

x = clnn.Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32), device=device)
y = clnn.Tensor(np.array([[0], [1], [1], [0]], np.float32), device=device)
optimizer = optim.Adam(model, learning_rate=0.03)

optimizer.zero_grad()
loss = clnn.binary_cross_entropy_with_logits(model(x), y)
loss.backward()
optimizer.step()
```

Passing `device=clnn.Device.opencl()` creates GPU-resident tensors; `.numpy()`, `.item()`,
and `.grad` copy their current values back to NumPy. The package is fully type-marked and
requires Python 3.9+ and NumPy.

See [the architecture](docs/ARCHITECTURE.md), [the usage guide](USAGE.md), and [migration notes](docs/MIGRATION.md).

## Scope

Tensors are contiguous `float32` values on a CPU or OpenCL GPU. The current surface intentionally excludes higher-order derivatives, arbitrary-stride views, and mixed precision. Unsupported shapes and cross-device operations fail explicitly.

Licensed under the [MIT License](LICENSE).
