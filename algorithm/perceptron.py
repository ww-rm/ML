import numpy as np

# ndarray的行数为实例特征个数, 列数为实例总数


def perceptron_base(train_x, train_y, yita):
    """感知机的原始算法

    参数: 
        x: np.ndarray类型, 每一行是一个样本
        y: 类型格式与x相同, 值取+1或-1
        yita: 学习率, n取小于等于1的正数

    返回:
        二元组(w, b, f)
        w为一维特征向量
        b为偏置常数
        f是模型函数
    """

    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=float)

    # 获得特征数与样本数
    total_num = train_x.shape[0]
    feature_num = train_x.shape[1]

    #w, b初始化为0
    w = np.zeros((feature_num, ))
    b = 0.0

    # 选取一个实例, index是当前的实例下标
    index = 0
    count = 0
    while index < total_num:
        if train_y[index] * (w @ train_x[index] + b) <= 0:
            w += yita*train_y[index]*train_x[index]
            b += yita*train_y[index]
            index = 0
        else:
            index += 1

        # 防止不收敛
        count += 1
        if count % 10000000 == 0:
            print("感知机迭代次数超过10000000次")

    def f(x):
        x = np.array(x)
        return np.sign(w @ x + b)

    return (w, b, f)


def perceptron_dual(train_x, train_y, yita):
    """感知机的对偶算法

    参数: 
        x: np.array类型, 每一行是一个样本
        y: 类型格式与x相同, 值取+1或-1
        yita: 学习率, n取小于等于1的正数

    返回:
        二元组(a, b, f)
        a为...
        b为偏置常数
        f是模型函数
    """

    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=float)

    # 获得特征数与样本数
    total_num = train_x.shape[0]
    feature_num = train_x.shape[1]

    # Gram 矩阵, 每次查第index列
    gram_table = train_x @ train_x.T

    #a, b初始化为0
    a = np.zeros((total_num, ))
    b = 0.0

    # 选取一个实例, index是当前的实例下标
    index = 0
    count = 0
    while index < total_num:
        if train_y[index] * (np.multiply(a, train_y) @ gram_table[..., index] + b) <= 0:
            # print(index, end=', ')
            a[index] += yita
            b += yita*train_y[index]
            index = 0
        else:
            index += 1

        # 防止不收敛
        count += 1
        if count % 10000000 == 0:
            print("感知机迭代次数超过10000000次")

    def f(x):
        x = np.array(x)
        return np.sign(np.multiply(a.T, train_y) @ train_x @ x + b)

    return (a.T, b, f)


class Perceptron:
    def __init__(self):
        self.total_num = None
        self.feature_num = None

    def fit(self, train_x, train_y, yita=0.01):
        """
        参数: 
            x: np.ndarray类型, 每一行是一个样本
            y: 类型格式与x相同, 值取+1或-1
            yita: 学习率, n取小于等于1的正数
        """

        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)

        # 获得特征数与样本数
        self.total_num = train_x.shape[0]
        self.feature_num = train_x.shape[1]

        #w, b初始化为0
        self.w = np.zeros((self.feature_num, ))
        self.b = 0.0

        # 选取一个实例, index是当前的实例下标
        index = 0
        count = 0
        while index < self.total_num:
            if train_y[index] * (self.w @ train_x[index] + self.b) <= 0:
                self.w += yita*train_y[index]*train_x[index]
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
