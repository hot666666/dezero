# Deeplearning_From_Scratch_3

## memo

---

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

---

## step14

- Variable의 동일 모듈에서 재사용하기 위해선 해당 변수의 grad를 축적해야한다
- cleargrad()는 이전에 backward()로 누적된 grad를 초기화시킨다
- in-place연산과 복사

  ```python
  class Variable:
      ...
      def backward(self):
          ...
          for gx, x in zip(gxs, f.inputs):
              if x.grad is None:
                  x.grad = gx
              else:
                  x.grad = x.grad + gx  # 복사(새로생성)가 이루어진다.
                  # x.grad += gx  # in-place 연산이 이루어진다. -> 이전 grad에 더해져 계산됨
  ```

  일반 연산의 경우, 연산의 결과값이 새로운 id의 변수에 할당되지만

  in-place연산의 경우, 같은 id의 변수에 값이 덮어쓰기가 된다.

  gx는 처음에 gy고, 그래서 y.grad에 연산결과들과 이어지는 것과 같다

## step16

- generation -> x, f
- set_func와 heapq에서 square에러

## step17

- weakref

## step18

- class 변수로 flag를 만들어 backprop을 제어
- contextlib.contextmanager에 setattr, getattr로 동적으로 flag(속성)을 처리
- 이를 return하는 함수로 만들어 with no_grad(): 구문이 탄생

  요약하면 with no_grad구문에선 Fucntion내 흐름을 제어하는 flag값을 바꾸고, 그 밖에선 원래대로 되돌림

  이때 contextmanager가 사용되며, 속성값은 getattr과 setattr로 처리한다.

## step23

- dezero

  - \_\_init\_\_.py
  - core_simple.py
    - Config
    - Funtion -> overloads
    - Variable

- python을 통해 파일 실행시 **file**이라는 전역변수가 정의된다.
- globals()는 현재 스코프의 전역변수를 리턴한다.

  ```python
      if '__file__' in globals():
          import os
          import sys
          sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
  ```

  pip install로 패키지를 설치한다면 파이썬 검색경로에 자동으로 추가되어 이런작업을 할 필요 없다.

## step25

- brew install graphviz
- dot [dot file] -T png -o [png파일 이름]

## step26

- 문자열 포맷

  나중에 문자열에 있던 {}에 변수값들이 들어감

  ```python
    dot_var = '{} [label={}, color=orange, style=filled]\n'
    v = ...
    name = ...
    dot_var.format(id(v), name)
  ```

## step28, 29

- 경사하강법 최적화 -> learning rate 필요, 미분값과 lr를 곱해 업데이트
- 뉴턴방법 최적화 -> 식을 테일러급수로 표현 후 2차미분값을 통해 업데이트

## step33_35

- [ ] 고차미분
- [ ] function.F
