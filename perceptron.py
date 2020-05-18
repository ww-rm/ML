import numpy as np
import json


class Perceptron:
    """二分类感知机"""

    def __init__(self):
        self.total_num = None
        self.feature_num = None

    def fit(self, train_x, train_y, iter_times, yita):
        """
        参数:
            train_x: csr格式稀疏矩阵
            train_y: ndarray, 值取+1或-1
            iter_times: 迭代次数
            yita: 学习率, n取小于等于1的正数
        """

        # 获得特征数与样本数
        self.total_num = train_x.shape[0]
        self.feature_num = train_x.shape[1]

        # Gram 矩阵, 每次查第index列
        print('calculate gram table...')
        gram_table = train_x.dot(train_x.T).toarray()

        # w, b初始化为0
        self.a = np.zeros(self.total_num)
        self.b = 0.0

        print('begin train...')
        # 迭代iter_times次
        index = 0
        count = 0  # 用来加快训练速度
        for _ in range(iter_times):
            if train_y[index] * ((self.a*train_y).dot(gram_table[..., index]) + self.b) <= 0:
                self.a[index] += yita
                self.b += yita*train_y[index]
                # print(self.b)
                count = 0
            else:
                count += 1
            # 如果已经满足所有的点都分开了
            if count == self.total_num:
                break
            # 判断下标是否需要回到0
            index += 1
            if index >= self.total_num:
                index %= self.total_num

        self.w = (self.a*train_y)*(train_x)  # type:ndarray
        # print(self.w, self.b)
        return True

    def predict(self, x):
        return self.w*x.T + self.b


class PerceptronC:
    """多分类感知机"""

    def __init__(self):
        self.total_num = None
        self.feature_num = None
        self.labels = None
        self.classifier = Perceptron()
        self.model = None

    def fit(self, train_x, train_y, iter_times=100000, yita=0.01):
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

        # 逐个训练
        print('begin each train...')
        for label in self.labels:
            current_y = np.ones(self.total_num)
            for index, y in enumerate(train_y):
                # 把非当前类的换成 -1
                if y != label:
                    current_y[index] = -1

            print(f'label {label}...')
            self.classifier.fit(train_x, current_y, iter_times, yita)
            print(f'label {label} done...')

            # 保存当前类别的 w 和 b
            self.model[str(label)] = [self.classifier.w, self.classifier.b]

        print('each train done...')

        return True

    def predict(self, x):
        # 从所有类别中选出值最大的那个
        result = ('', float('-inf'))
        for label, w_b in self.model.items():
            self.classifier.w, self.classifier.b = w_b
            y = self.classifier.predict(x)
            if y > result[1]:
                result = (label, y)

        return int(result[0])

    def predictAll(self, samples):
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
