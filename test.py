from perceptron import *
import numpy as np
import random

train_x = np.mat(
    [[1, 1],
     [2, 1],
     [1, 2],
     [5, 5],
     [6, 5],
     [5, 6]]
).T

train_y = np.mat(
    [[1,1,1,-1,-1,-1]]
)


w, b = perceptron_base(train_x, train_y, 0.1)
print(w,b)

a, b = perceptron_dual(train_x, train_y, 0.1)
print(a,b)
print(np.multiply(a.T, train_y)@train_x.T)