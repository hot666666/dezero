import numpy as np
from step11_12_13_func_args import as_array, add as old_add, Variable as OldVariable


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
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for gx, x in zip(gxs, f.inputs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # 부록A

                if x.creator is not None:
                    funcs.append(x.creator)

    def clear_grad(self):
        # x.grad가 None이 아니면 x.grad+=gx가 실행되므로, 다른 함수에 사용될 때에는 x.grad를 None으로 만들어야 한다.
        self.grad = None


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gys):
        return gys, gys


def add(x0, x1):
    return Add()(x0, x1)


if __name__ == '__main__':
    x = OldVariable(np.array(10))
    y = old_add(x, x)
    y.backward()
    print(y.data, x.grad == 2)  # 동일 변수에 대해 grad가 누적되지 않음 -> False

    x = Variable(np.array(10))
    y = add(x, x)
    y.backward()
    print(y.data, x.grad == 2)

    y = add(add(x, x), x)
    y.backward()
    print(x.grad == 3)  # Variable x는 y=add(x,x)로 사용되어 grad가 남아있음 -> False

    x.clear_grad()
    y = add(add(x, x), x)
    y.backward()
    print(x.grad == 3)
