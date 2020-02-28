import numpy as np

# mat的行数为实例特征个数, 列数为实例总数


def perceptron_base(train_x, train_y, yita):
    """感知机的原始算法

    参数: 
        x: np.mat类型, 行数为特征数, 列数为样本数
        y: 类型格式与x相同, 值取+1或-1
        yita: 学习率, n取小于等于1的正数

    返回:
        二元组(w, b), 均为np.mat类型
        w为特征向量矩阵
        b为偏置常数
    """

    train_x = np.mat(train_x)
    train_y = np.mat(train_y)

    # 获得特征数与样本数
    feature_num = train_x.shape[0]
    total_num = train_x.shape[1]

    #w, b初始化为0
    w = np.zeros((1, feature_num))
    b = np.zeros((1, 1))

    # 选取一个实例, index是当前的实例下标
    index = 0
    count = 0
    while index < total_num:
        if train_y[..., index] * w @ train_x[..., index] + b <= 0:
            # print(index, end=', ')
            w += yita*train_y[..., index]*train_x[..., index].T
            b += yita*train_y[..., index]
            index = 0
        else:
            index += 1

        # 防止不收敛
        count += 1
        if count > 1000000:
            print("感知机迭代次数超过1000000次")

    return (w, b)


def perceptron_dual(train_x, train_y, yita):
    """感知机的对偶算法

    参数: 
        x: np.mat类型, 行数为特征数, 列数为样本数
        y: 类型格式与x相同, 值取+1或-1
        yita: 学习率, n取小于等于1的正数

    返回:
        二元组(a, b), 均为np.mat类型
        a为...
        b为偏置常数
    """

    train_x = np.mat(train_x)
    train_y = np.mat(train_y)

    # 获得特征数与样本数
    # feature_num = train_x.shape[0]
    total_num = train_x.shape[1]

    # Gram 矩阵, 每次查第index列
    gram_table = train_x.T @ train_x

    #a, b初始化为0
    a = np.zeros((1, total_num))
    b = np.zeros((1, 1))

    # 选取一个实例, index是当前的实例下标
    index = 0
    count = 0
    while index < total_num:
        if train_y[..., index] @ np.multiply(a, train_y) @ gram_table[..., index] + b <= 0:
            # print(index, end=', ')
            a[..., index] += yita
            b += yita*train_y[..., index]
            index = 0
        else:
            index += 1

        # 防止不收敛
        count += 1
        if count > 1000000:
            print("感知机迭代次数超过1000000次")

    return (a.T, b)
