# Deeplearning_From_Scratch_3

## step7-8

재귀 -> 반복

```python
class Variable:
    ...
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
```

```python
class Variable:
    ...
    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
```

## step9

numpy에서 0차원 ndarray 인스턴스를 사용하여 계산 시, 타입이 달라진다

### example

- y = np.array(1.0)\*\*2
- type(y)는 numpy.ndarray->numpy.float64

## step10

python -m unittest discover [tests파일명]

- unittest.TestCase를 상속한 클래스에 test\_메소드를 통해 진행
- unittest
  - assertEqual, assertTrue(numpy.allclose -> bool)
