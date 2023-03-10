import numpy as np

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from dezero import Variable
    import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(create_graph=True)
print(x.grad)


x2 = Variable(np.random.randn(1, 2, 3))
print(x2)
y2 = x2.reshape(2, 3)
print(y2)


x3 = Variable(np.random.randn(2, 3))
y3_1 = x3.transpose()
y3_2 = x3.T
print(y3_1)
print(y3_2)
