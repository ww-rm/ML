from knn import *
import numpy as np

train_x = np.array([
    [1, 1],
    [1, 2],
    [2, 2],
    [2, 3],
    [2, 1],
    [3, 2],
    [3, 1],
    [1, 3],
    [3, 3]
]).T

train_y = np.array([
    3, 1, 3, 1, 2, 1, 3, 1, 2
])

# print(train_x[:,1].argsort())
# train_x = train_x[train_x[:,1].argsort()]
# print(train_x)

# train_input = np.concatenate(
#     train_x, train_y.reshape(1, 9)
# ).T

# print(knn_base(train_x, train_y, [1.51, 3], 1))

tree = KdTree()
tree.make_tree(train_x, train_y)
a = tree.search_tree((1.50001,2.49999), 3)
print(a)