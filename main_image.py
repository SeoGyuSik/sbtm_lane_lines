import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import pickle
from Binary_threshold import thresh_binary
from Prsp_transform import warper
import Identify_lines as Id
from collections import deque


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.recent_fit = deque(maxlen=3)


a = Line()
fit1 = np.array([1, 2, 3])
a.recent_fit.append(fit1)
print(a.recent_fit)
fit2 = np.array([4, 5, 6])
a.recent_fit.append(fit2)
print(a.recent_fit)
fit3 = np.array([7, 8, 9])
a.recent_fit.append(fit3)
print(a.recent_fit)
fit4 = np.array([10, 11, 12])
a.recent_fit.append(fit4)
print(a.recent_fit)
print(len(a.recent_fit))


def get_mean_fit():
    s0 = 0
    s1 = 0
    s2 = 0
    for fit in a.recent_fit:
        s0 += fit[0]
        s1 += fit[1]
        s2 += fit[2]
    m0 = s0 / len(a.recent_fit)
    m1 = s1 / len(a.recent_fit)
    m2 = s2 / len(a.recent_fit)
    return np.array([m0, m1, m2])


print(get_mean_fit())








