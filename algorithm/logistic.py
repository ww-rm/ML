import numpy as np


class Logistic:
    def __init__(self):
        self.total_num = None
        self.feature_num = None

    def _sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def fit(self, train_x, train_y, yita=0.0001):
        """
        参数:
            x: np.ndarray类型, 行数为特征数, 列数为样本数
            y: 类型格式与x相同, 值取+1或0
            yita: 学习率, n取小于等于1的正数
        """

        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)

        # 获得特征数与样本数
        self.feature_num = train_x.shape[0]
        self.total_num = train_x.shape[1]

        # w合并b之后, 初始化为0
        self.w = np.zeros((self.feature_num + 1, ))

        # 在x样本特征的末尾加入一个常数项1
        train_x = np.concatenate((train_x, np.ones((1, self.total_num))))

        # print(train_x)
        # print(train_y)

        # 迭代一段时间
        flag = True
        while flag:
            # print(self.w)
            flag = False
            h = self._sigmoid(self.w@train_x)
            error = train_y - h
            new_w = self.w + yita*train_x@error

            # 迭代到一定的精度之后退出
            for num in (new_w-self.w).flat:
                if np.fabs(num) > 1e-6:
                    self.w = new_w
                    flag = True
                    break
            
            

        self.b = self.w[-1]
        self.w = self.w[0:-1]

        return True

    def predict(self, x):
        """
        返回值:
            二元组(y, prob)
            第一个值为概率大于0.5的分类结果
            第二个值为概率值
        """

        x = np.array(x)
        # print(x, self.w, self.b)
        prob_0 = 1.0/(1+np.exp(self.w @ x + self.b))
        return (0, prob_0) if prob_0 > 0.5 else (1, 1-prob_0)
