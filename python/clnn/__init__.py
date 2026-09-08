from __future__ import annotations

import sys as _sys

from . import _clnn as _native
from ._clnn import (
    Device,
    DeviceType,
    KernelProfile,
    OpenCLRuntimeStatistics,
    Tensor,
    TensorStatistics,
    avg_pool2d,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    conv2d,
    cross_entropy,
    exp,
    grad_enabled,
    inspect,
    leaky_relu,
    log,
    matmul,
    max_pool2d,
    mean,
    mse_loss,
    no_grad,
    numel,
    opencl_available,
    pow,
    relu,
    reshape,
    sigmoid,
    shape_string,
    softmax,
    statistics,
    sum,
    tanh,
    transpose,
)

__version__ = _native.__version__
nn = _native.nn
optim = _native.optim
data = _native.data
explain = _native.explain

_sys.modules[f"{__name__}.nn"] = nn
_sys.modules[f"{__name__}.optim"] = optim
_sys.modules[f"{__name__}.data"] = data
_sys.modules[f"{__name__}.explain"] = explain

__all__ = [
    "Device",
    "DeviceType",
    "KernelProfile",
    "OpenCLRuntimeStatistics",
    "Tensor",
    "TensorStatistics",
    "avg_pool2d",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "conv2d",
    "cross_entropy",
    "data",
    "exp",
    "explain",
    "grad_enabled",
    "inspect",
    "leaky_relu",
    "log",
    "matmul",
    "max_pool2d",
    "mean",
    "mse_loss",
    "nn",
    "no_grad",
    "numel",
    "opencl_available",
    "optim",
    "pow",
    "relu",
    "reshape",
    "sigmoid",
    "shape_string",
    "softmax",
    "statistics",
    "sum",
    "tanh",
    "transpose",
]
