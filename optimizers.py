from typing import Union, Callable

import numpy as np

from models import Model
from layers import Layer
from variable import Parameter


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks: list[Callable] = []

    def setup(self, target: Union[Layer, Model]):
        self.target: Union[Layer, Model] = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param: Parameter):
        raise NotImplementedError

    def add_hook(self, f: Callable):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super(SGD, self).__init__()
        self.lr = lr

    def update_one(self, param: Parameter):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super(MomentumSGD, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param: Parameter):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
