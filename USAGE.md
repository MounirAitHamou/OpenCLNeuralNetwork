# Usage

## Tensors and gradients

Create a leaf tensor with `requires_grad=true`, compose ordinary tensor operations, then call `backward()` on a scalar result:

```cpp
clnn::Tensor x({1.0F, 2.0F, 3.0F}, {3}, true);
auto loss = clnn::mean(x * x);
loss.backward();
// x.grad() == {2/3, 4/3, 2}
```

Gradients accumulate on leaf tensors. Call `zero_grad()` directly or through an optimizer before the next step. A non-scalar output requires an explicit gradient argument. The graph is released after backward unless `retain_graph=true` is supplied.

Use `clnn::NoGradGuard` around evaluation and parameter-independent inference.

## GPU execution

```cpp
if (!clnn::opencl_available()) throw std::runtime_error("OpenCL GPU required");
auto gpu = clnn::Device::opencl(platform_index, gpu_index);
auto x_gpu = x.to(gpu);
model.to(gpu);
// Optional explicit barrier; data() and grad() already wait when needed.
gpu.synchronize();
```

All operands of an operation must share a device. GPU values, gradients, and optimizer moments
retain OpenCL buffers across forward, backward, and optimizer operations. Reading `data()`/`grad()`,
inspection statistics, state serialization, and host-side value validation synchronize the
requested values to the host.

OpenCL kernel launches are asynchronous and execute in order. CLNN flushes submitted work so the
device may run while the host continues. Use `Device::synchronize()` only when host code needs an
explicit completion barrier without reading a tensor; blocking tensor reads synchronize
automatically.

Profiling and allocator statistics are available directly on the GPU device:

```cpp
gpu.set_profiling(true);
auto output = model(x);
for (const auto& kernel : gpu.profile())
    std::cout << kernel.operation << ": " << kernel.total_milliseconds << " ms\n";
const auto runtime = gpu.runtime_statistics();
gpu.set_profiling(false);
```

`profile()` synchronizes the queue and resets collected events by default. Buffer pooling and kernel
caching remain active when profiling is disabled; `clear_memory_pool()` releases currently reusable
buffers.

## Modules

Modules own leaf parameters and construct a graph in `forward()`. `Sequential` owns its child modules and prefixes state names with their stable index. Inputs to `Linear` are `[batch, features]`; inputs to `Conv2d` are contiguous NCHW tensors.

`MaxPool2d` and `AvgPool2d` accept NCHW input. A zero stride selects the kernel size. Average
pooling divides by the number of valid input elements, excluding padding. Dimension-wise
`sum(input, dimension, keep_dimensions)` and `mean(...)` support arbitrary tensor ranks.

Call `model.train()` before optimization and `model.eval()` for inference. The mode propagates
through `Sequential`: `Dropout` samples inverted-dropout masks only during training, and
`BatchNorm` uses batch statistics during training and its persisted running statistics during
evaluation.

`LayerNorm` normalizes one or more trailing dimensions. `GlobalAvgPool2d` converts NCHW input to
`[batch, channels]`, and `Residual` wraps any shape-preserving child module and returns
`input + child(input)`.

```cpp
model.add(std::make_unique<clnn::nn::BatchNorm>(channels, 1.0e-5F, 0.1F, true, gpu))
     .add(std::make_unique<clnn::nn::ReLU>())
     .add(std::make_unique<clnn::nn::MaxPool2d>(std::array<std::size_t, 2>{2, 2}))
     .add(std::make_unique<clnn::nn::Dropout>(0.25F, 42));
model.eval();
```

## Explainability

`clnn::explain` returns attribution tensors with the same shape and device as the explained input.
An attribution describes sensitivity or contribution under the fitted model; it is not evidence of
a causal relationship. Select logits when possible: post-softmax class probabilities couple every
class and can have saturated gradients.

```cpp
const auto target = clnn::explain::Target{3};
auto saliency = clnn::explain::saliency(model, input, target);
auto input_gradient = clnn::explain::input_x_gradient(model, input, target);
auto baseline = clnn::Tensor::zeros(input.shape(), false, {}, input.device());
auto integrated =
    clnn::explain::integrated_gradients(model, input, target, baseline, 64);
auto smoothed = clnn::explain::smoothgrad(model, input, target, 50, 0.1F, 42);
```

- Saliency is the absolute gradient of the selected output with respect to the input.
- Input x Gradient multiplies the signed gradient by the input.
- Integrated Gradients multiplies the input-to-baseline difference by an average path gradient.
  CLNN uses a right-endpoint Riemann sum at `k / steps` for `k = 1..steps`.
- SmoothGrad averages saliency over deterministic Gaussian-perturbed inputs. A fixed seed repeats
  the same noise stream.

For an output of rank two or greater, dimension zero is treated as the batch and `Target::index`
addresses the flattened remaining dimensions. `Target{class_index}` sums that class output across
the batch, producing an attribution for every input sample. `Target{class_index, batch_index}`
selects one sample, leaving other samples' attribution zero. Rank-one outputs are unbatched; scalar
outputs accept only target zero. Invalid indices fail before backward.

An Integrated Gradients baseline must exactly match the input shape. Zeros are a common starting
point, but a domain-meaningful neutral input is usually better; the baseline should represent an
absence of features for the application. A baseline on another device is copied to the input
device once.

Explanation calls temporarily use evaluation mode and restore the prior top-level mode. They use
fresh input leaves and a selected vector-Jacobian backward pass, so user inputs and existing model
parameter gradients are untouched. They temporarily enable graph recording even when called inside
a `NoGradGuard`, then restore the caller's previous setting. CPU inputs remain on CPU. OpenCL inputs, path samples,
gradients, and attribution arithmetic remain GPU-resident; SmoothGrad noise is generated by the
seeded host RNG and uploaded, while reading `data()`/`.numpy()` remains an explicit synchronization.

Python exposes the same interface as `from clnn import explain`. For a `[C, H, W]` attribution,
a dependency-free channel reduction is `np.abs(attr.numpy()).sum(axis=0)`; plotting is intentionally
left to the application. Python accepts either `target=3, batch_index=0` or
`target=explain.Target(3, 0)`.

```python
import numpy as np
import clnn
from clnn import explain

baseline = clnn.Tensor.zeros(x.shape, device=x.device)
attr = explain.integrated_gradients(
    model, x, target=3, baseline=baseline, steps=64
)
heatmap = np.abs(attr.numpy()[0]).sum(axis=0)  # [N,C,H,W] -> first [H,W]
```

Modules also support general forward observation hooks. Keep the returned handle alive, and call
`remove()` (or destroy it) to unregister. Hooks receive the module, input, and output Tensor handles,
do not alter the output, and run for nested modules invoked through `module(input)`. Retaining an
output Tensor in a hook is safe and permits reading its gradient after backward without forcing a
GPU synchronization during the hook itself.

## Saving parameters

```cpp
clnn::nn::save_state_dict(model, "model.clnn");
clnn::nn::load_state_dict(model, "model.clnn");
```

Loading validates every parameter and persistent-buffer name and shape before changing the module,
preventing partially loaded models. Batch-normalization running statistics are state-dict entries.
Use `save_checkpoint` / `load_checkpoint` to persist a supported sequential architecture plus state,
and `optim::save_state_dict` / `optim::load_state_dict` to resume optimizer moments exactly.

## Optimizer groups and schedules

Different parameter sets can use independent learning rates and weight decay. Schedulers update all
groups:

```cpp
clnn::optim::SGD optimizer(clnn::optim::ParameterGroups{{
    {backbone.parameters(), 1.0e-3F, 1.0e-4F},
    {head.parameters(), 1.0e-2F, 0.0F},
}});
clnn::optim::CosineAnnealingLR schedule(optimizer, epochs);
// Call after each epoch.
schedule.step();
```

`ExponentialLR` and `StepLR` are also available. Optimizer state files retain parameter-group
settings and remain load-compatible with the earlier single-group format.

## Benchmarks

Configure with `CLNN_BUILD_BENCHMARKS=ON` (the default), build Release, and run
`clnn_benchmarks`. Each timing includes one queue synchronization around a batch of iterations,
so it measures asynchronous throughput without forcing a barrier after every kernel. GPU runs also
report aggregated event timings and cache/pool counters. `clnn_benchmarks --check` runs a shorter
suite with broad upper bounds intended to catch catastrophic CI regressions.

## Data

`data::load_csv_numerical` loads selected named columns. `data::load_cifar10_batch` reads standard
CIFAR-10 binary batches in NCHW order. `data::split` and `data::DataLoader` provide deterministic
partitions, shuffling, variable final batches, optional drop-last behavior, and direct batch placement
on a GPU. Pass `prefetch=true` to prepare the next batch, including its host-to-device transfer, on a
background task while the caller works on the current batch.
