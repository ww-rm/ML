from perceptron import *
import numpy as np
import random
import matplotlib.pyplot as plt

N = 50
bias = 0

train_x = np.concatenate(
    (np.random.rand(2, N)*-10-1, np.random.rand(2, N)*(10)+1), 1)

train_y = np.array(
    [1] * (N+bias) + [-1] * (N-bias)
)


result_base = perceptron_base(train_x, train_y, 1e-5)
result_dual = perceptron_dual(train_x, train_y, 10)
# print(result_base)
# print(result_dual)
# print(np.multiply(result_dual[0].T, train_y)@train_x.T)


def f1(x):
    w = result_base[0]
    b = result_base[1]
    return -(w[0]*x+b)/w[1]


def f2(x):
    w = np.multiply(result_dual[0].T, train_y)@train_x.T
    b = result_dual[1]
    return -(w[0]*x+b)/w[1]


line_x = [-10, 10]
plt.plot(line_x, list(map(f1, line_x)), color='yellow')
plt.plot(line_x, list(map(f2, line_x)), color='green')
plt.scatter(train_x[0][:N+bias], train_x[1][:N+bias], color='blue')
plt.scatter(train_x[0][N+bias:], train_x[1][N+bias:], color='red')

plt.show()
