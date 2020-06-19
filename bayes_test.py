import json

import scipy.sparse
from matplotlib import pyplot
from sklearn.metrics import classification_report, f1_score

from bayes import NaiveBayes

if __name__ == "__main__":
    classfier = NaiveBayes()
    train_x = scipy.sparse.load_npz('./data/pre-train/tfidf.npz')
    with open('./data/pre-train/labels.json', encoding='utf8') as f:
        train_y = json.load(f)
    test_x = scipy.sparse.load_npz('./data/pre-test/tfidf.npz')
    with open('./data/pre-test/labels.json', encoding='utf8') as f:
        test_y = json.load(f)
    
    # 这一段代码用来调参的
    # tongji = []
    # for i in range(1, 101):
    #     print(f'{i/1000}...')
    #     classfier.fitText(train_x, train_y, i/1000)
    #     predict_y = classfier.predictTextAll(test_x)
    #     results = f1_score(test_y, predict_y, average='macro')
    #     print(results)
    #     tongji.append((i/1000, results))
    
    # pyplot.xticks([i/1000 for i in range(1, 101)])
    # pyplot.scatter([i[0] for i in tongji], [i[1] for i in tongji])
    # pyplot.plot()
    # pyplot.show()

    classfier.fitText(train_x, train_y, 0.024)
    # # # classfier.saveModel('./model/bayes_tfidf_0024.json') # count: 0.071 tfidf: 0.024
    # classfier.readModel('./model/bayes_tfidf_0024.json')
    predict_y = classfier.predictTextAll(test_x)
    results = classification_report(test_y, predict_y)
    print(results)
