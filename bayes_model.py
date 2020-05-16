from bayes import NaiveBayes
import json
from pprint import pprint
from matplotlib import pyplot


with open('./data/words.dict', encoding='utf8') as f:
    wordsdict = json.load(f)
with open('./data/categories.json') as f:
    categories = json.load(f)
print('length:', len(wordsdict))


def loadSparseData(filepath):
    """返回x和y"""
    data_x = []
    data_y = []

    # print(labels)
    # 读取每一个类
    for index, category in enumerate(categories):
        with open(filepath+'/'+category, encoding='utf8') as f:
            sparse = json.load(f)
            data_x.extend(sparse)
            data_y.extend([index]*len(sparse))
            del sparse

    return (data_x, data_y)

def evaluateBayes(test_y, predict_y):
    # 分类统计结果
    # true, false, total, precision=true/(true+false), recall=true/total, macro-f1
    # results = {'title':['true', 'false', 'total', 'precision', 'recall', 'macro-f1']}
    results = {i: [0, 0, 0, 0.0, 0.0, 0.0] for i in categories}
    for i, j in zip(test_y, predict_y):
        # if i == 2:
        #     print(j, end=' ')
        results[categories[i]][2] += 1
        if i == j:
            results[categories[i]][0] += 1
        else:
            results[categories[j]][1] += 1
    for i in categories:
        results[i][3] = round(results[i][0]/(results[i][0]+results[i][1]), 3)
        results[i][4] = round(results[i][0]/results[i][2], 3)
        results[i][5] = round(2*results[i][3]*results[i][4]/(results[i][3]+results[i][4]), 3)
    results['total'] = [
        sum([i[0] for i in results.values()]),
        sum([i[1] for i in results.values()]),
        sum([i[2] for i in results.values()]),
        sum([i[3] for i in results.values()])/len(categories),
        sum([i[4] for i in results.values()])/len(categories),
        sum([i[5] for i in results.values()])/len(categories)
    ]

    return results

if __name__ == "__main__":
    classfier = NaiveBayes(len(wordsdict))
    train_x, train_y = loadSparseData('./data/sparse-train')
    test_x, test_y = loadSparseData('./data/sparse-test')
    
    # 这一段代码用来调参的
    # tongji = []
    # for i in range(1, 101):
    #     print(f'{i/1000}...')
    #     classfier.fitText(train_x, train_y, i/1000)
    #     predict_y = classfier.predictTextAll(test_x)
    #     results = evaluateBayes(test_y, predict_y)
    #     tongji.append((i/1000, results.get('total')[-1]))    
    # pprint(tongji)
    # pyplot.xticks([i/1000 for i in range(1, 101)])
    # pyplot.scatter([i[0] for i in tongji], [i[1] for i in tongji])
    # pyplot.plot()
    # pyplot.show()

    classfier.fitText(train_x, train_y, 0.071)
    # classfier.saveModel()
    # classfier.readModel()
    predict_y = classfier.predictTextAll(test_x)
    results = evaluateBayes(test_y, predict_y)
    pprint(results)