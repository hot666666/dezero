import numpy as np
from step2_function import Function, Square
from step1_variable import Variable


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


if __name__ == '__main__':
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)
