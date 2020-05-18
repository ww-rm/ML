import json
import os
import re
import string

import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
import scipy.io
import scipy.sparse
from sklearn import datasets

a = scipy.sparse.coo_matrix(
    [[1, 2, 0],
     [4, 0, 6],
     [0, 8, 9]]
).tocsc()

b = scipy.sparse.coo_matrix(
    [[1, 2, 0],
     [4, 0, 6],
     [0, 8, 9]]
).tocsr()

c = scipy.sparse.coo_matrix(
    [[1, 0, 0, 0, 0, 0, 9, 5],
     [1, 0, 0, 0, 0, 0, 9, 5],
     [1, 0, 0, 0, 0, 0, 9, 5],
     [1, 0, 0, 0, 0, 0, 9, 5],
     [1, 0, 0, 0, 0, 0, 9, 5],
     [1, 0, 0, 0, 0, 0, 9, 5],
     [1, 0, 0, 0, 0, 0, 9, 5],
     [1, 0, 0, 0, 0, 0, 9, 5]]
).toarray()
# print(c)
e = np.ones(8)
print((e*c[0]))

# d = [1, 0, 0, 0, 0, 0, 9, 5]
# print('begin...')
# print(a.dot(a.T).toarray())
