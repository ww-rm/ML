import json
import numpy as np


class Perceptron:
    """二分类感知机"""

    def __init__(self):
        self.total_num = None
        self.feature_num = None
        self.a = None
        self.w = None
        self.b = None

    def fit(self, train_x, train_y, gram_table, iter_times):
        """
        参数:
            train_x: csr格式稀疏矩阵
            train_y: ndarray, 值取+1或-1
            gram_table: 提前计算出来的训练样本内积矩阵
            iter_times: 迭代次数
            yita: 学习率, n取小于等于1的正数
        """

        # 获得特征数与样本数
        self.total_num = train_x.shape[0]
        self.feature_num = train_x.shape[1]

        # a, b初始化为0
        self.a = np.zeros(self.total_num)
        self.b = 0.0

        print('begin train...')
        # 迭代iter_times次
        for _ in range(iter_times):
            # 随机选点
            index = np.random.randint(0, self.total_num)
            if train_y[index] * ((self.a*train_y).dot(gram_table[..., index]) + self.b) <= 0:
                self.a[index] += 1
                self.b += 1*train_y[index]

        self.w = (self.a*train_y)*(train_x)  # type:ndarray
        # print(self.w, self.b)
        return True

    def predict(self, x):
        """
        x是行向量
        """

        return self.w*x.T + self.b


class PerceptronC:
    """多分类感知机"""

    def __init__(self):
        self.total_num = None
        self.feature_num = None
        self.labels = None
        self.classifier = Perceptron()
        self.model = None

    def fit(self, train_x, train_y, iter_times=100000):
        """
        参数:
            x: coo稀疏矩阵
            y: ndarray, 标签值
            iter_times: 迭代次数
            yita: 学习率, n取小于等于1的正数
        """

        # 获得特征数与样本数标签数
        self.total_num = train_x.shape[0]
        self.feature_num = train_x.shape[1]
        self.labels = sorted(set(train_y))

        self.model = {str(label): None for label in self.labels}

        train_x = train_x.tocsr()

        # Gram 矩阵, 每次查第index列
        print('calculate gram table...')
        gram_table = train_x.dot(train_x.T).toarray()

        # 逐个训练
        print('begin each train...')
        for label in self.labels:
            current_y = np.ones(self.total_num)
            for index, y in enumerate(train_y):
                # 把非当前类的换成 -1
                if y != label:
                    current_y[index] = -1

            print(f'label {label}...')
            self.classifier.fit(train_x, current_y, gram_table, iter_times)
            print(f'label {label} done...')

            # 保存当前类别的 w 和 b
            self.model[str(label)] = [self.classifier.w, self.classifier.b]

        print('each train done...')

        return True

    def predict(self, x):
        """
        接受的是csr稀疏矩阵
        """

        # 从所有类别中选出值最大的那个
        result = ('', float('-inf'))
        for label, w_b in self.model.items():
            self.classifier.w, self.classifier.b = w_b
            y = self.classifier.predict(x)
            if y > result[1]:
                result = (label, y)

        return int(result[0])

    def predictAll(self, samples):
        """
        samples是coo稀疏矩阵
        """

        print(f'predict {samples.shape[0]} samples...')
        results = []

        for sample in samples.tocsr():
            results.append(self.predict(sample))

        print('predict done...')
        return results

    def saveModel(self, filename='PerceptronC.json'):
        print('save PerceptronC model...')
        model = {
            label: [self.model[label][0].tolist(), self.model[label][1]]
            for label in self.model
        }
        with open(filename, 'w') as f:
            json.dump(model, f)
        print('save done...')

    def readModel(self, filename='PerceptronC.json'):
        print('read PerceptronC model...')
        with open(filename) as f:
            model = json.load(f)
        self.model = {
            label: [np.array(model[label][0]), model[label][1]] for label in model
        }
        print('read done...')
