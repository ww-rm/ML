from perceptron import *
from aaaaaaaaaaaa import Perceptron
import numpy as np
import random
import matplotlib.pyplot as plt

N = 50
bias = 0

train_x = np.concatenate(
    (np.random.rand(2, N)*(10)+10, np.random.rand(2, N)*(10)+1), 1).T

train_y = np.array(
    [-1] * (N+bias) + [1] * (N-bias)
)

# print(train_x, train_y)

classifier =  Perceptron()
classifier.fit(train_x, train_y, 5000000, yita=1e-3)

print(classifier.w, classifier.b)

result_base = perceptron_base(train_x, train_y, 1e-3)
# result_dual = perceptron_dual(train_x, train_y, 1e-3)
print(result_base)


def f1(x):
    w = result_base[0]
    b = result_base[1]
    return -(w[0]*x+b)/w[1]


def f2(x):
    w = classifier.w
    b = classifier.b
    return -(w[0]*x+b)/w[1]


line_x = [0, 50]
plt.plot(line_x, list(map(f1, line_x)), color='yellow')
plt.plot(line_x, list(map(f2, line_x)), color='green')
plt.scatter(train_x[:N+bias, 0], train_x[:N+bias, 1], color='blue')
plt.scatter(train_x[N+bias:, 0], train_x[N+bias:, 1], color='red')

plt.show()
