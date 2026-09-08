# Architecture

Each `Tensor` is a small shared handle to contiguous CPU storage or an OpenCL buffer, optional gradient storage, a mutation version, and an optional graph node. An operation creates its output eagerly and, when gradient recording is enabled, stores parent handles plus a vector-Jacobian-product callback.

`Tensor::backward()` discovers the reachable graph, produces a topological ordering, seeds the output gradient, and invokes callbacks in reverse order. Shared parents naturally accumulate contributions. Parent versions captured during the forward pass detect invalid optimizer or user mutations before derivatives are evaluated.

Modules contain parameters but no backward implementation. Losses are tensor operations. Optimizers only read leaf gradients and update leaf values under `NoGradGuard`.

Explainability reuses this same reverse traversal with a selected output-gradient mask. During that
pass, leaf accumulation is restricted to a fresh copy of the explained input; intermediate
gradients still flow normally, while parameter gradient buffers are neither cleared nor updated.
Forward hooks are dispatched by `Module::operator()` from a lifetime-safe shared registry. Hook
handles remove registrations using weak ownership, so destroying a module before its handle is safe.

The CPU kernels are deliberately straightforward reference implementations. OpenCL kernels implement the same forward and backward semantics and are continuously checked against that oracle. The OpenCL loader is resolved at runtime, keeping the public headers and build independent of any vendor SDK.

GPU tensor values, gradients, parameters, and optimizer moments stay in OpenCL buffers between
operations. Backward callbacks pass device-resident gradient tensors, and branch contributions are
accumulated by an in-place OpenCL kernel. Routine backward and optimizer steps perform no buffer
reads. Host transfers are reserved for `data()`/`grad()`/statistics, serialization, device-state
migration, and the few operations that perform host-side value validation.

OpenCL kernels are submitted to an in-order command queue and flushed without waiting. Dependencies
between operations therefore remain implicit in queue order while the host can continue submitting
work. Blocking buffer reads and writes wait as required, and `Device::synchronize()` provides an
explicit full-queue barrier. Kernel objects are cached by entry-point name instead of being rebuilt
for each launch. Released buffers enter an exact-size pool after their last in-order event completes,
which avoids a global wait while preventing premature reuse. Runtime shutdown drains the queue before
releasing events, pooled buffers, kernels, and other resources.

Profiling is opt-in because it creates one retained event per launch. A profiling-enabled command
queue records device start/end timestamps; `Device::profile()` finishes the queue, aggregates those
events by kernel, and releases them. Runtime statistics expose allocation, reuse, pool, and kernel-
cache counts independently of profiling.

Dimension reductions use workgroup-local tree reductions, and pooling has paired OpenCL forward and
vector-Jacobian-product kernels. Higher-level batch and layer normalization compose axis reductions
and broadcast operations, so their training graphs use the same CPU/OpenCL autograd path rather than
a special backward pass.
Dropout creates a seeded inverted-dropout mask on the input device and becomes an identity in eval
mode. Module mode and device migration propagate recursively through `Sequential`; persistent
batch-normalization statistics migrate and serialize as named buffers.

Matrix multiplication uses 16-by-16 workgroups and local-memory tiles. Global dimensions are
rounded to tile boundaries while kernel guards preserve arbitrary matrix shapes. The benchmark
target times batches between explicit synchronization points to expose queue throughput. Convolution
similarly stages an input tile and one output channel's weights in local memory when the device has
enough local storage, with the general direct kernel retained as a fallback.
