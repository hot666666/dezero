import numpy as np
from step1_variable import Variable


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


if __name__ == '__main__':
    f = Square()
    x = Variable(np.array(10))
    y = f(x)
    print(type(y))
    print(y.data)
