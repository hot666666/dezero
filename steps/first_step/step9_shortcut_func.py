import numpy as np
from step8_auto_backprop import Square, Exp


def Square(x):
    # return Square()(x)
    f = Square()
    return f(x)


def Exp(x):
    # return Exp()(x)
    f = Exp()
    return f(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            # self.data와 같은 타입으로 만들기 위해 np.ones_like 사용
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


if __name__ == '__main__':
    x = np.array(1.0)
    y = x**2
    print(type(x), x.ndim)
    print(type(y), y.ndim)
