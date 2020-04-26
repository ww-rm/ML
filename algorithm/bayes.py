import numpy as np


class NaiveBayes:
    """离散型朴素贝叶斯"""
    def __init__(self):
        self.total_num = None
        self.feature_num = None
        self.values = None
        self.prior_prob = None
        self.conditional_prob = None

    def fit(self, train_x, train_y, lam=1):
        """
        参数: 
            train_x: 训练集, 列向量形式
            train_y: 样本类别值, 一维数组
            lam: 贝叶斯估计时lamda的值, 默认 1
        """

        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)

        self.feature_num = train_x.shape[0]  # 特征数
        self.total_num = train_x.shape[1]  # 样本总数
        self.values = []  # values保存所有维度的取值集合, 包括y
        self.prior_prob = {}  # 类别y的先验概率
        self.conditional_prob = {}  # 在某一个y下的x某一分量取某一值的条件概率 self.conditional_prob[y][xi][xij]

        train_input = np.concatenate((train_x, train_y.reshape(1, self.total_num))).T

        # 获取所有维度的取值集合
        for i in train_input.T:
            self.values.append(np.unique(i))

        # 数据预处理分组
        data = {}
        for i in train_input:
            if data.get(i[-1]) is None:
                data[i[-1]] = [i[:-1]]
            else:
                data[i[-1]].append(i[:-1])

        for i in self.values[-1]:
            self.prior_prob[i] = (len(data[i]) + lam) / (self.total_num + len(self.values[-1])*lam)

        # 计算conditional_prob
        for i in self.values[-1]:  # y的取值
            i_list = []  # 某类别下的各维度概率字典集合
            current_x = np.array(data[i]).T  # 当前类别的实例集合
            for j in range(self.feature_num):  # 维度的取值
                j_dict = {}
                for k in self.values[j]:
                    j_dict[k] = (sum(current_x[j] == k) + lam) / (len(data[i]) + len(self.values[j])*lam)  # 计算某维度上的某个值的概率
                i_list.append(j_dict)  # 添加维度各取值概率字典
            self.conditional_prob[i] = i_list  # 添加各维度的列表

        return True

    def predict(self, x):
        x = np.array(x, dtype=float)
        # 对于x, 计算每个类别y的后验概率
        x_prob = {}
        for i in self.values[-1]:
            tmp = self.prior_prob[i]
            for j in range(self.feature_num):
                tmp *= self.conditional_prob[i][j][x[j]]
            x_prob[i] = tmp

        # print(x_prob)
        # 选出概率最大的类别
        result = max(x_prob.items(), key=lambda v: v[1])[0]

        return result