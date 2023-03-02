import numpy as np
import math

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from dezero import Variable


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


if __name__ == '__main__':
    x = Variable(np.array(np.pi/4))
    y = my_sin(x)
    y.backward()
    print(y.data, np.sin(x.data))
    print(x.grad, np.cos(x.data))
