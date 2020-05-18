from perceptron import PerceptronC
import json
from pprint import pprint
from matplotlib import pyplot
from sklearn.metrics import f1_score, classification_report
import scipy.io


if __name__ == "__main__":
    classfier = PerceptronC()
    train_x = scipy.io.mmread('./data/pre-train/tfidf.mtx')
    with open('./data/pre-train/labels.json', encoding='utf8') as f:
        train_y = json.load(f)
    test_x = scipy.io.mmread('./data/pre-test/tfidf.mtx')
    with open('./data/pre-test/labels.json', encoding='utf8') as f:
        test_y = json.load(f)

    # # 这一段代码用来调参的
    # tongji = []
    # for i in range(1, 101):
    #     print(f'{i/100}...')
    #     classfier.fit(train_x, train_y, yita=i/100)
    #     predict_y = classfier.predictAll(test_x)
    #     results = f1_score(test_y, predict_y, average='macro')
    #     print(results)
    #     tongji.append((i/100, results))
    # pprint(tongji)
    # pyplot.xticks([i/100 for i in range(1, 101)])
    # pyplot.scatter([i[0] for i in tongji], [i[1] for i in tongji])
    # pyplot.plot()
    # pyplot.show()

    # classfier.fit(train_x, train_y, 100000, yita=0.01)
    # # classfier.saveModel() # 10w, 0.01
    classfier.readModel()
    predict_y = classfier.predictAll(test_x)
    results = classification_report(test_y, predict_y)
    print(results)
