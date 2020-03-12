from perceptron import *
import numpy as np
import random
import matplotlib.pyplot as plt

print(dir(np.array([1,2,3])))

x = np.array(
    [[1, 2],
     [3, 4],
     [5, 6]]
).T

y = np.array(
    [1, 10, 2]
).reshape(1, 3)

z = np.concatenate((x, y)).T
p = np.array(x)
p[0][0] = 111

np.c
print(x, p)
