if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from dezero import Variable

import numpy as np


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


if __name__ == '__main__':
    x1 = Variable(np.array(2.0))

    lr = 0.001
    iters = 100

    for i in range(iters):
        if i % 10 == 0:
            print(x1)

        y = f(x1)
        # 이전계산의 기울기를 지우고 backward()
        x1.cleargrad()
        y.backward()

        x1.data -= lr * x1.grad

    print('---')
    x2 = Variable(np.array(2.0))

    iters = 10

    for i in range(iters):
        print(x2)

        y = f(x2)
        x2.cleargrad()
        y.backward()

        x2.data -= x2.grad / gx2(x2.data)
