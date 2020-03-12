from knn import *
import numpy as np
import matplotlib.pyplot as plt

train_x = np.random.rand(2, 100)*100
train_y = np.random.randint(1, 6, (100, ))

# print(train_x, train_y)

x = np.random.rand(2)*100

# print(x)

tree = Knn()
tree.fit(train_x, train_y)
result = tree.predict(x, 5)
print(result)

plt.scatter(x[0], x[1], color='black', marker='p')
for i in range(100):
    if train_y[i] == 1:
        plt.scatter(train_x[0][i], train_x[1][i], color='blue', marker='.')
    if train_y[i] == 2:
        plt.scatter(train_x[0][i], train_x[1][i], color='red', marker='.')
    if train_y[i] == 3:
        plt.scatter(train_x[0][i], train_x[1][i], color='yellow', marker='.')
    if train_y[i] == 4:
        plt.scatter(train_x[0][i], train_x[1][i], color='green', marker='.')
    if train_y[i] == 5:
        plt.scatter(train_x[0][i], train_x[1][i], color='cyan', marker='.')

plt.show()
