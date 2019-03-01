import numpy as np
from collections import deque


def get_mean_fit(a):
    s0 = 0
    s1 = 0
    s2 = 0
    for fit in a:
        s0 += fit[0]
        s1 += fit[1]
        s2 += fit[2]
    m0 = s0 / len(a)
    m1 = s1 / len(a)
    m2 = s2 / len(a)

    return np.array([m0, m1, m2], dtype='float')


a = deque(maxlen=5)
print(len(a))
c = [2.20226688e-04,  -2.40052530e-01,   4.07356742e+02]
c2 = [2.20226688e-04,  -2.40052530e-01,   4.07356742e+01]
a.append(c)
a.append(c2)
a.append(c)
a.append(c2)
a.append(c2)
a.append(c2)
b = get_mean_fit(a)
print(a)
print(c)
print(b)
print(len(a))
