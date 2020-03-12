from bayes import *
import numpy as np

train_x = np.array(
    [[1, 4],
     [1, 5],
     [1, 5],
     [1, 4],
     [1, 4],
     [2, 4],
     [2, 5],
     [2, 5],
     [2, 6],
     [2, 6],
     [3, 6],
     [3, 5],
     [3, 5],
     [3, 6],
     [3, 6]]
).T

train_y = np.array(
    [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
)

classifier = NaiveBayes()
classifier.fit(train_x, train_y)
r = classifier.predict([2, 4])

print(r)
# out: -1
