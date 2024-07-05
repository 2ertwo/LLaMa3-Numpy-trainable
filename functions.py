import numpy as np
from typing import Union
from variable import Variable
from config import Config
from tools import as_variable, as_array, plot_dot_graph, reshape_sum_backward
from tools import sum_to as raw_sum_to
from cuda import get_array_module
import weakref


class Function:
    def __call__(self, *inputs: Union[np.ndarray, Variable]) -> Union[list[Variable], Variable]:
        inputs: list[Variable] = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]

        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.outputs = [weakref.ref(output) for output in outputs]
            self.inputs = inputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gys: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = y * gy
        return gx


def square(x: Variable):
    return Square()(x)


def exp(x: Variable):
    return Exp()(x)


# class XtoX(Function):
#     # y = x^x
#     # lny = xlnx
#     # y'/y = 1+lnx
#     # y' = y(1+lnx)
#     def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
#         return x ** x
#
#     def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
#         y = self.outputs[0]()
#         x = self.inputs[0]
#         xp = get_array_module(x)
#         gx = gy * (y * (1 + xp.log(x)))
#         return gx


class Log(Function):
    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        xp = get_array_module(x)
        return xp.log(x)

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        x = self.inputs[0]
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x, y):
    y = as_array(y, get_array_module(x))
    return Add()(x, y)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray]:
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        x0, x1 = self.inputs
        gx0, gx1 = gy * x1, gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


def mul(x, y):
    y = as_array(y, get_array_module(x))
    return Mul()(x, y)


class Neg(Function):
    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        return -x

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray]:
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x, y):
    y = as_array(y, get_array_module(x))
    return Sub()(x, y)


def rsub(x, y):
    y = as_array(y, get_array_module(x))
    return Sub()(y, x)


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray]:
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


def div(x, y):
    y = as_array(y, get_array_module(x))
    return Div()(x, y)


def rdiv(x, y):
    y = as_array(y, get_array_module(x))
    return Div()(y, x)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        y = x ** self.c
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        x = self.inputs[0]
        gx = self.c * x ** (self.c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


class Sin(Function):
    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        xp = get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        x = self.inputs[0]
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        xp = get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        x = self.inputs[0]
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        xp = get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


# class Transpose(Function):
#     def forward(self, x: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
#         y = np.transpose(x)
#         return y
#
#     def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
#         return transpose(gy)
#
#
# def transpose(x):
#     return Transpose()(x)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis: Union[tuple[int, ...], int, None], keepdims: bool):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        self.x_shape = x.shape
        xp = get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        self.x_shape = x.shape
        y = raw_sum_to(x, self.shape)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray) -> tuple[np.ndarray]:
        if x.ndim <= 2 and W.ndim <= 2:
            y = x.dot(W)
        else:
            y = x @ W
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        x, W = self.inputs
        gx = matmul(gy, W.transpose(([i for i in range(W.ndim - 2)] + [-1, -2])))
        gW = matmul(x.transpose(([i for i in range(x.ndim - 2)] + [-1, -2])), gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        x0, x1 = self.inputs
        diff: Variable = x0 - x1
        gx0: Variable = gy * diff * (2. / len(diff))
        gx1: Variable = -gx0
        return gx0, gx1


def mean_squared_error(x, y):
    return MeanSquaredError()(x, y)


class Linear(Function):
    def forward(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray]:
        y = x.dot(W)
        if b is not None:
            y += b

        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.transpose(([i for i in range(W.ndim - 2)] + [-1, -2])))
        gW = matmul(x.transpose(([i for i in range(x.ndim - 2)] + [-1, -2])), gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        # y = 1 / (1 + exp(-x))
        xp = get_array_module(x)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        y = x[self.slices]
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        x = self.inputs[0]
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


def get_item(x, slices):
    return GetItem(slices)(x)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        xp = get_array_module(gy)
        gx = xp.zeros(self.in_shape)
        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx: np.ndarray) -> Union[tuple[Variable, ...], Variable]:
        return get_item(ggx, self.slices)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        xp = get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis=axis)(x)


class Cat(Function):
    def __init__(self, axis: int = 0):
        self.axis = axis

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        xp = get_array_module(xs[0])
        z = xp.concatenate(xs, axis=self.axis)
        return z

    def backward(self, gy: Variable) -> Union[tuple[Variable, ...], Variable]:
        inputs = self.inputs
        gx = []
        start_idx = 0
        for x in inputs:
            end_idx = start_idx + x.shape[self.axis]
            indices = [slice(None)] * gy.ndim
            indices[self.axis] = slice(start_idx, end_idx)
            gx.append(gy[tuple(indices)])
            start_idx = end_idx

        return tuple(gx)


def cat(inputs, axis=0):
    return Cat(axis=axis)(*inputs)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        xp = get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        x = self.inputs[0]
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()

    return Variable(as_array(acc))


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if Config.train:
        xp = get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


class Stack(Function):
    def __init__(self, axis: int = 0):
        self.axis = axis

    def forward(self, *xs: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        xp = get_array_module(xs[0])
        self.x_shape = xs[0].shape
        self.x_num = len(xs)
        y = xp.stack(xs, axis=self.axis)
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        gx = []
        for i in range(self.x_num):
            indices = [slice(None)] * gy.ndim
            indices[self.axis] = slice(i, i + 1)
            gx.append(gy[tuple(indices)].reshape(self.x_shape))
        return tuple(gx)


def stack(inputs, axis=0):
    return Stack(axis=axis)(*inputs)


if __name__ == '__main__':
    # def goldstein(x, y):
    #     z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
    #         (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    #     return z
    #
    #
    # x = Variable(np.array(1.0))
    # y = Variable(np.array(1.0))
    # z = goldstein(x, y)
    # z.backward()
    # print(z)
    # print(x.grad, y.grad)
    #
    # plot_dot_graph(z, verbose=False)

    # x = Variable(np.array(1.0))
    # y = tanh(x)
    # x.name = 'x'
    # y.name = 'y'
    # y.backward(create_graph=True)
    # iters = 6
    #
    # for i in range(iters):
    #     gx = x.grad
    #     x.cleargrad()
    #     gx.backward(create_graph=True)
    #
    # gx = x.grad
    # gx.name = 'gx' + str(iters + 1)
    # plot_dot_graph(gx, verbose=False)

    # np.random.seed(0)
    # x = np.random.rand(100, 1)
    # y = 5 + 2 * x + np.random.rand(100, 1)
    #
    # W = Variable(np.zeros((1, 1)))
    # b = Variable(np.zeros(1))
    #
    #
    # def predict(x):
    #     y = matmul(x, W) + b
    #     return y
    #
    #
    # def mean_squared_error(x0, x1):
    #     diff = x0 - x1
    #     return sum(diff ** 2) / len(diff)
    #
    #
    # lr = 0.1
    # iters = 100
    #
    # for i in range(iters):
    #     y_pred = predict(x)
    #     loss = mean_squared_error(y, y_pred)
    #     W.cleargrad()
    #     b.cleargrad()
    #     loss.backward()
    #     W.data -= lr * W.grad.data
    #     b.data -= lr * b.grad.data
    #     print(W, b, loss)
    # a = Variable(np.array([i for i in range(20)]).reshape(4, 5))
    # b = Variable(np.array([i for i in range(20, 40)]).reshape(4, 5))
    # c = Variable(np.array([i for i in range(40, 60)]).reshape(4, 5))
    #
    # aaa = 2
    # raw_shape = a.shape
    # print(raw_shape[:aaa] + (1,) + raw_shape[aaa:])
    #
    # a1 = a.reshape(raw_shape[:aaa] + (1,) + raw_shape[aaa:])
    # b1 = b.reshape(raw_shape[:aaa] + (1,) + raw_shape[aaa:])
    # c1 = c.reshape(raw_shape[:aaa] + (1,) + raw_shape[aaa:])
    #
    # d = cat((a1, b1, c1), axis=aaa)
    #
    # d1 = d[1:, :, :] * 3
    # d2 = d[:1, :, :] * 5
    #
    # dd = cat((d1, d2), axis=0)
    #
    # dd.backward()
    # print(dd)
    # print(dd.shape)
    # print(a.grad)
    # print(b.grad)
    # print(c.grad)
    #
    # a2 = Variable(np.array([i for i in range(20)]).reshape(4, 5))
    # b2 = Variable(np.array([i for i in range(20, 40)]).reshape(4, 5))
    # c2 = Variable(np.array([i for i in range(40, 60)]).reshape(4, 5))
    #
    # dd2 = Stack(axis=aaa)(a2, b2, c2)
    #
    # d11 = dd2[1:, :, :] * 3
    # d22 = dd2[:1, :, :] * 5
    #
    # ddd = cat((d11, d22), axis=0)
    # ddd.backward()
    #
    # print(ddd)
    # print(a2.grad)
    # print(b2.grad)
    # print(c2.grad)

    pass
