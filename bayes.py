import numpy as np


def naive_bayes(train_x, train_y, x, lam=1):
    """朴素贝叶斯算法, 用于离散型变量, 采用贝叶斯估计

    输入: 
        train_x: 训练集, 列向量形式
        train_y: 样本类别值, 一维数组
        x: 待预测的样本的特征向量
        lam: 贝叶斯估计时lamda的值, 默认 1

    输出:
        x的类别
    """

    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=float)
    x = np.array(x, dtype=float)
    # 获得特征数与样本数, 以及数据预处理
    feature_num = train_x.shape[0]
    total_num = train_x.shape[1]
    train_input = np.concatenate((train_x, train_y.reshape(1, total_num))).T

    # values保存所有维度的取值集合, 包括y
    values = []
    for i in train_input.T:
        values.append(np.unique(i))

    # 数据预处理分组
    data = {}
    for i in train_input:
        if data.get(i[-1]) is None:
            data[i[-1]] = [i[:-1]]
        else:
            data[i[-1]].append(i[:-1])

    # 类别y的先验概率
    prior_prob = {}
    for i in values[-1]:
        prior_prob[i] = (len(data[i]) + lam) / (total_num + len(values[-1])*lam)

    # 在某一个y下的x某一分量取某一值的条件概率
    # conditional_prob[y][xi][xij]
    conditional_prob = {}

    # 计算conditional_prob
    for i in values[-1]:  # y的取值
        i_list = []  # 某类别下的各维度概率字典集合
        current_x = np.array(data[i]).T  # 当前类别的实例集合
        for j in range(feature_num):  # 维度的取值
            j_dict = {}
            for k in values[j]:
                j_dict[k] = (sum(current_x[j] == k) + lam) / (len(data[i]) + len(values[j])*lam)  # 计算某维度上的某个值的概率
            i_list.append(j_dict)  # 添加维度各取值概率字典
        conditional_prob[i] = i_list  # 添加各维度的列表

    # 对于x, 计算每个类别y的后验概率
    x_prob = {}
    for i in values[-1]:
        tmp = prior_prob[i]
        for j in range(feature_num):
            tmp *= conditional_prob[i][j][x[j]]
        x_prob[i] = tmp

    # print(x_prob)
    # 选出概率最大的类别
    result = max(x_prob.items(), key=lambda v: v[1])
    return result
