import numpy as np
import random
import matplotlib.pyplot as plt

x = np.array(
    [[1, 2],
     [3, 4],
     [5, 6]]
).T

y = np.array(
    [1, 10, 2]
).reshape(1, 3)

z = np.concatenate((x, y), 0)

print(x, y)
print(np.exp(z))

