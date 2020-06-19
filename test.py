import json
import os
import numpy as np

import scipy.sparse
from sklearn.metrics import classification_report

from bayes import NaiveBayes
from perceptron import PerceptronC
from preprocess import myTokenizer, preProcess

if __name__ == "__main__":
    a = scipy.sparse.coo_matrix(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    ).tocsr()

    b = np.array(
        [9, 9, 9]
    )
    
    sss = 1
    print(a.indptr)
    print(sss)
