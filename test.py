from perceptron import *
import numpy as np
import random
import matplotlib.pyplot as plt

x = np.array([1, -4, 5])

y = np.array([1, 10, 2])

z = np.concatenate((x, y))

z[0] = 999
z[0:3].sort()
print(z)
c = [7,6,5,4,3]
c[0:3].sort()
print(c)