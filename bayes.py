import json
import math



class NaiveBayes:
    """离散型朴素贝叶斯"""

    def __init__(self, dictlength):
        self.feature_num = dictlength
        self.prior_prob = None
        self.conditional_prob = None

    def fitText(self, train_x, train_y, lam=1):
        """接受稀疏矩阵的学习

        train_x的形式: [
            [[项号, 值], ...], ...
        ]
        train_y的形式: [值, ...], 要纯数字标签
        """
        
        print('train text NaiveBayes...')
        self.total_num = len(train_y)  # 样本数
        self.prior_prob = {}  # 类别y的先验概率
        self.conditional_prob = {}  # 在某一个y下的所有单词概率 self.conditional_prob[y][index]

        # 计算类别的先验概率
        for label in train_y:
            if self.prior_prob.get(str(label)) is None:
                self.prior_prob[str(label)] = 1
            else:
                self.prior_prob[str(label)] += 1
        for label in self.prior_prob:
            self.prior_prob[label] = (self.prior_prob[label]+lam)/(self.total_num+len(self.prior_prob)*lam)
        # print(self.prior_prob)

        # 计算每个类每个每个单词出现概率
        # 准备一下数据格式
        for label in self.prior_prob:
            self.conditional_prob[label] = [0]*self.feature_num

        # {'类别一': [单词0出现的次数/概率, 单词1出现的次数/概率, ...]
        #  '类别二': [...]
        # }
        # 统计训练集中每个类别单词出现的次数, 并统计每个类别的总词数
        for sample, label in zip(train_x, train_y):
            for index, word_count in sample:
                # 累加次数
                self.conditional_prob[str(label)][index] += word_count

        # 转换成概率
        for label in self.conditional_prob:
            words_count = sum(self.conditional_prob[label])
            for index in range(self.feature_num):
                self.conditional_prob[label][index] = (self.conditional_prob[label][index]+lam)/(words_count+self.feature_num*lam)

        # print(self.conditional_prob)
        print('train done...')


    def predictText(self, x):
        """返回值是数字结果"""

        result = ('', float('-inf'))

        # 对每一个类别计算概率
        for label in self.prior_prob:
            tmp = math.log(self.prior_prob[label])
            for index, word_count in x:
                tmp += word_count*math.log(self.conditional_prob[label][index])
            # print(tmp)
            if tmp > result[1]:
                result = (label, tmp)

        return int(result[0])

    def predictTextAll(self, samples):
        print(f'predict {len(samples)} samples...')
        results = []

        for sample in samples:
            results.append(self.predictText(sample))

        print('predict done...')
        return results

    def saveModel(self, filename='NaiveBayes.json'):
        print('save NaiveBayes model...')
        with open(filename, 'w') as f:
            json.dump(
                {
                    'prior_prob': self.prior_prob,
                    'conditinal_prob': self.conditional_prob
                }, f)
        print('save done...')

    def readModel(self, filename='NaiveBayes.json'):
        print('read NaiveBayes model...')
        with open(filename) as f:
            model = json.load(f)
            self.prior_prob = model.get('prior_prob')
            self.conditional_prob = model.get('conditinal_prob')
        print('read done...')

    def fit(self, train_x, train_y):
        pass