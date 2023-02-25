import numpy as np
from first_step import step9_shortcut_func as step9


a = step9.Variable(np.array(2))
print(a.data)  # step9의 import step8때문에 오류발생
