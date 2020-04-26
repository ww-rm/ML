import numpy as np

class Logistic:
    def __init__(self):
        self.total_num = None
        self.feature_num = None

    def fit(self, train_x, train_y, yita=0.01):
        """
        参数: 
            x: np.ndarray类型, 行数为特征数, 列数为样本数
            y: 类型格式与x相同, 值取+1或-1
            yita: 学习率, n取小于等于1的正数
        """

        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)

        # 获得特征数与样本数
        self.feature_num = train_x.shape[0]
        self.total_num = train_x.shape[1]

        #w, b初始化为0
        self.w = np.zeros((self.feature_num, ))
        self.b = 0.0

        # 选取一个实例, index是当前的实例下标
        index = 0
        count = 0
        while index < self.total_num:
            if train_y[index] * (self.w @ train_x[..., index] + self.b) <= 0:
                self.w += yita*train_y[index]*train_x[..., index].T
                self.b += yita*train_y[index]
                index = 0
            else:
                index += 1

            # 防止不收敛
            count += 1
            if count % 10000000 == 0:
                print("感知机迭代次数超过10000000次")

        return True

    def predict(self, x):
        x = np.array(x)
        return np.sign(self.w @ x + self.b)
