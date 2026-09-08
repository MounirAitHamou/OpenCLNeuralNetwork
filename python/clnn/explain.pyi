from . import Tensor
from .nn import Module
from typing import overload

class Target:
    def __init__(self, index: int, batch_index: int | None = ...) -> None: ...
    index: int
    batch_index: int | None

@overload
def saliency(model: Module, input: Tensor, target: Target) -> Tensor: ...
@overload
def saliency(model: Module, input: Tensor, target: int,
             batch_index: int | None = ...) -> Tensor: ...
@overload
def input_x_gradient(model: Module, input: Tensor, target: Target) -> Tensor: ...
@overload
def input_x_gradient(model: Module, input: Tensor, target: int,
                     batch_index: int | None = ...) -> Tensor: ...
@overload
def integrated_gradients(model: Module, input: Tensor, target: Target, baseline: Tensor,
                         steps: int = ...) -> Tensor: ...
@overload
def integrated_gradients(model: Module, input: Tensor, target: int, baseline: Tensor,
                         steps: int = ..., batch_index: int | None = ...) -> Tensor: ...
@overload
def smoothgrad(model: Module, input: Tensor, target: Target, samples: int = ...,
               noise_stddev: float = ..., seed: int = ...) -> Tensor: ...
@overload
def smoothgrad(model: Module, input: Tensor, target: int, samples: int = ...,
               noise_stddev: float = ..., seed: int = ...,
               batch_index: int | None = ...) -> Tensor: ...
