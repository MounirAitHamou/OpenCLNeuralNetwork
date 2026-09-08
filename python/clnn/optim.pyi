from os import PathLike
from typing import overload

from .nn import Module

class ParameterGroup:
    def __init__(self, model: Module, learning_rate: float,
                 weight_decay: float = ...) -> None: ...
    learning_rate: float
    weight_decay: float

class Optimizer:
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    @property
    def parameter_group_count(self) -> int: ...
    def learning_rate(self, group: int = ...) -> float: ...
    def set_learning_rate(self, learning_rate: float, group: int = ...) -> None: ...

class SGD(Optimizer):
    @overload
    def __init__(self, model: Module, learning_rate: float, momentum: float = ...,
                 weight_decay: float = ...) -> None: ...
    @overload
    def __init__(self, parameter_groups: list[ParameterGroup],
                 momentum: float = ...) -> None: ...

class Adam(Optimizer):
    @overload
    def __init__(self, model: Module, learning_rate: float = ..., beta1: float = ...,
                 beta2: float = ..., epsilon: float = ..., weight_decay: float = ...,
                 decoupled_weight_decay: bool = ...) -> None: ...
    @overload
    def __init__(self, parameter_groups: list[ParameterGroup], beta1: float = ...,
                 beta2: float = ..., epsilon: float = ...,
                 decoupled_weight_decay: bool = ...) -> None: ...

class AdamW(Optimizer):
    @overload
    def __init__(self, model: Module, learning_rate: float = ..., beta1: float = ...,
                 beta2: float = ..., epsilon: float = ...,
                 weight_decay: float = ...) -> None: ...
    @overload
    def __init__(self, parameter_groups: list[ParameterGroup], beta1: float = ...,
                 beta2: float = ..., epsilon: float = ...) -> None: ...

class LRScheduler:
    def step(self) -> None: ...
    @property
    def steps(self) -> int: ...

class ExponentialLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float) -> None: ...

class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int,
                 gamma: float = ...) -> None: ...

class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, maximum_steps: int,
                 minimum_learning_rate: float = ...) -> None: ...

def save_state_dict(optimizer: Optimizer, path: str | PathLike[str]) -> None: ...
def load_state_dict(optimizer: Optimizer, path: str | PathLike[str]) -> None: ...
