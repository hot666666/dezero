import numpy as np
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dezero import Variable
# from dezero.utils import get_dot_graph


def _dot_var(v, verbose=True):
    dot_var = '{} [label={}, color=orange, style=filled]\n'
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ':'
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label={}, color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    txt += '{} -> {}\n'.format(id(f), id(f.outputs))
    for x in f.inputs:
        txt += '{} -> {}\n'.format(id(x), id(f))
        if x.creator is not None:
            txt += _dot_func(x.creator)
    return txt


def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
            for x in f.inputs:
                if x.creator is not None:
                    add_func(x.creator)
    add_func(output.creator)
    _dot_var(output, verbose)
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'


if __name__ == '__main__':
    x0 = Variable(np.array(1.0))
    x0.name = 'x0'
    x1 = Variable(np.array(1.0))
    x1.name = 'x1'
    t = x0 + x1
    y = x0 * x1
    z = t * y
    txt = get_dot_graph(z, verbose=False)
    print(txt)
