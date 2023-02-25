import numpy as np
from step11_12_13_func_args import as_array
import heapq


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation - 1  # for heapq

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:  # y1 -> f1(y1), f2(y1)인 경우?
                try:
                    print(f.__dict__)
                    heapq.heappush(funcs, (f.generation, f))
                    seen_set.add(f)
                except Exception as e:
                    print(e)

        add_func(self.creator)

        while funcs:
            _, f = heapq.heappop(funcs)
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for gx, x in zip(gxs, f.inputs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

    def clear_grad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    a = Square()(x)
    y = Add()(Square()(a), Square()(a))

    y.backward()
    print(y.data == 32)
    print(x.grad == 64)  # x = self.inputs[0].data므로 x=2에서의 미분값이 출력
