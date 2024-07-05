import numpy as np
from typing import TYPE_CHECKING
from config import using_config, Config

try:
    import cupy

    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)
if TYPE_CHECKING:
    from functions import Function


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.name = name
        self.generation = 0

    def set_creator(self, func: "Function"):
        self.creator: "Function" = func
        self.generation = func.generation + 1

    def backward(self, retain_grad: bool = False, create_graph=False):
        from cuda import get_array_module
        if self.grad is None:
            xp = get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs: list[Function] = []
        seen_set: set[Function] = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f: "Function" = funcs.pop()
            xs = f.inputs
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(xs, gxs):
                    if x.grad is not None:
                        x.grad = x.grad + gx
                    else:
                        x.grad = gx

                    if x.creator is not None:
                        add_func(x.creator)
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None

    def cleargrad(self):
        self.grad = None

    def to_cpu(self):
        from cuda import as_numpy
        if self.data is not None:
            self.data = as_numpy(self.data)

    def to_gpu(self):
        from cuda import as_cupy
        if self.data is not None:
            self.data = as_cupy(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'Variable(' + p + ')'

    def __add__(self, other):
        from functions import add
        return add(self, other)

    def __radd__(self, other):
        from functions import add
        return add(self, other)

    def __mul__(self, other):
        from functions import mul
        return mul(self, other)

    def __rmul__(self, other):
        from functions import mul
        return mul(self, other)

    def __neg__(self):
        from functions import neg
        return neg(self)

    def __sub__(self, other):
        from functions import sub
        return sub(self, other)

    def __rsub__(self, other):
        from functions import rsub
        return rsub(self, other)

    def __truediv__(self, other):
        from functions import div
        return div(self, other)

    def __rtruediv__(self, other):
        from functions import rdiv
        return rdiv(self, other)

    def __pow__(self, power, modulo=None):
        from functions import pow
        return pow(self, power)

    def __getitem__(self, item):
        from functions import get_item
        return get_item(self, item)

    def reshape(self, *shape):
        from functions import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    # def transpose(self):
    #     from functions import transpose
    #     return transpose(self)
    #
    # @property
    # def T(self):
    #     from functions import transpose
    #     return transpose(self)

    def transpose(self, *axes):
        from functions import transpose
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return transpose(self, axes)

    @property
    def T(self):
        from functions import transpose
        return transpose(self)

    def sum(self, axis=None, keepdims=False):
        from functions import sum
        return sum(self, axis=axis, keepdims=keepdims)


class Parameter(Variable):
    pass


if __name__ == '__main__':
    # a1 = Variable(np.array([1.0, 3.0]))
    # a2 = Variable(np.array([1.0, 2.0]))
    # print(a1 + a2)
    pass
