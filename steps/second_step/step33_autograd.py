if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from dezero import Variable

import numpy as np

'''
def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)

gx = x.grad
gx.backward()
print(x.grad)


x2 = Variable(np.array(2.0))
y2 = f(x2)
y2.backward(create_graph=True)
print(x2.grad)

gx2 = x2.grad
x2.cleargrad()
gx2.backward()
print(x2.grad)

'''


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
