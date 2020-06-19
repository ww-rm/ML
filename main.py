import json
import os

import scipy.sparse
from sklearn.metrics import classification_report

from bayes import NaiveBayes
from perceptron import PerceptronC
from preprocess import myTokenizer, preProcess

if __name__ == "__main__":
    pass
    print('**********数据预处理**********')
    categories = os.listdir('./data/20news-bydate-train')
    preProcess('./data/20news-bydate-train', './data/pre-train', categories, myTokenizer)
    with open('./data/wordsdict.dict', encoding='utf8') as f:
        wordsdict = json.load(f)
    preProcess('./data/20news-bydate-test/', './data/pre-test', categories, myTokenizer, wordsdict)
    print('**********预处理结束**********')

    print('********读取预处理数据********')
    train_x = scipy.sparse.load_npz('./data/pre-train/tfidf.npz')
    with open('./data/pre-train/labels.json', encoding='utf8') as f:
        train_y = json.load(f)
    test_x = scipy.sparse.load_npz('./data/pre-test/tfidf.npz')
    with open('./data/pre-test/labels.json', encoding='utf8') as f:
        test_y = json.load(f)
    print('***********读取结束***********')
    
    print('**********感知机测试**********')
    classfier = PerceptronC()
    classfier.fit(train_x, train_y, 80000)
    # classfier.saveModel('./model/PerceptronC.json')
    # classfier.readModel('./model/PerceptronC.json')
    predict_y = classfier.predictAll(test_x)
    results = classification_report(test_y, predict_y)
    print(results)
    print('********感知机测试结束********')

    print('**********贝叶斯测试**********')
    classfier = NaiveBayes()
    classfier.fitText(train_x, train_y, 0.024)
    # classfier.saveModel('./model/bayes_tfidf_0024.json')
    # classfier.readModel('./model/bayes_tfidf_0024.json')
    predict_y = classfier.predictTextAll(test_x)
    results = classification_report(test_y, predict_y)
    print(results)
    print('********贝叶斯测试结束********')
