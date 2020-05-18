import json
import math


class NaiveBayes:
    """离散型朴素贝叶斯"""

    def __init__(self):
        self.feature_num = None
        self.total_num = None
        self.model = None

    def fitText(self, train_x, train_y, lam=0.1):
        """接受稀疏矩阵的学习

        train_x: coo稀疏矩阵
        train_y: [值, ...], 要纯数字标签
        """

        print('train text NaiveBayes...')
        self.total_num = train_x.shape[0]  # 样本数
        self.feature_num = train_x.shape[1]
        self.model = {str(label): [0, [0]*self.feature_num] for label in set(train_y)}
        # self.model = [先验概率, [单词一的概率, 单词二, ...]]

        # 计算类别的先验概率
        for label in train_y:
            self.model[str(label)][0] += 1
        for label in self.model:
            self.model[label][0] = (self.model[label][0]+lam)/(self.total_num+len(self.model)*lam)

        # 计算每个类每个每个单词出现概率
        # {'类别一': [xxx, [单词0出现的次数/概率, 单词1出现的次数/概率, ...]]
        #  '类别二': [...]
        # }
        # 统计训练集中每个类别单词出现的次数, 并统计每个类别的总词数
        for row, col, word_count in zip(train_x.row, train_x.col, train_x.data):
            self.model[str(train_y[row])][1][col] += word_count

        # 转换成概率
        for label in self.model:
            words_count = sum(self.model[label][1])
            for index in range(self.feature_num):
                self.model[label][1][index] = (self.model[label][1][index]+lam)/(words_count+self.feature_num*lam)

        # print(self.model)
        print('train done...')

    def predictText(self, x):
        """返回值是数字结果
        
        x:
            必须是csr稀疏矩阵,
            而且是单个样本
        """

        result = ('', float('-inf'))

        # 对每一个类别计算概率
        for label in self.model:
            probaility = math.log(self.model[label][0])
            for index, word_count in zip(x.indices, x.data):
                probaility += word_count*math.log(self.model[label][1][index])
            if probaility > result[1]:
                result = (label, probaility)

        return int(result[0])

    def predictTextAll(self, samples):
        """接受稀疏矩阵的预测

        samples:
            coo稀疏矩阵
        """

        print(f'predict {samples.shape[0]} samples...')
        results = []

        for sample in samples.tocsr():
            results.append(self.predictText(sample))

        print('predict done...')
        return results

    def saveModel(self, filename='NaiveBayes.json'):
        print('save NaiveBayes model...')
        with open(filename, 'w') as f:
            json.dump(self.model, f)
        print('save done...')

    def readModel(self, filename='NaiveBayes.json'):
        print('read NaiveBayes model...')
        with open(filename) as f:
            self.model = json.load(f)
        print('read done...')

    def fit(self, train_x, train_y):
        pass
