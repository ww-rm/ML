import numpy as np


def knn_base(train_x, train_y, x, k=1, p=2):
    """线性查找方式的knn算法

    输入:
        train_x: 训练集, 列向量形式
        train_y: 样本类别值, 一维数组
        x: 待预测的样本的特征向量
        k: k近邻
        p: 距离公式的范数
    输出:
        x的类别值, 一定在train_y中
    """
    # 用来保存距离
    distances = []

    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=float)
    x = np.array(x, dtype=float)

    # 获得特征数与样本数
    # feature_num = train_x.shape[0]
    total_num = train_x.shape[1]

    # 计算所有的距离
    for i in range(total_num):
        distances.append(
            (np.linalg.norm(x-train_x[..., i], ord=p), train_y[i])
        )

    # 按距离排序
    distances.sort()

    # (距离, 类别)
    k_neighbor = distances[0:k]

    # 计算各类数量
    neighbor_count = {}
    for neighbor in k_neighbor:
        if neighbor_count.get(neighbor[1]) is None:
            neighbor_count[neighbor[1]] = 1
        else:
            neighbor_count[neighbor[1]] += 1

    # 表决
    result = None
    max_count = 0
    for i in neighbor_count.items():
        if i[1] > max_count:
            result = i[0]
            max_count = i[1]

    return result


class KdTreeNode:
    """KdTree的节点结构体"""
    def __init__(self, depth=-1, pre_node=None):
        self.depth = depth
        self.pre_node = pre_node  # type: KdTreeNode
        self.left_node = None  # type: KdTreeNode
        self.right_node = None  # type: KdTreeNode
        self.vector = None
        self.category = None


class KdTree:
    def __init__(self):
        self.root = None

    def make_tree(self, train_x, train_y):
        """生成KdTree, 在使用search_tree方法之前需要调用一次
        
        输入:
            train_x: 训练集, 列向量形式
            train_y: 样本类别值, 一维数组
        """
        
        train_x = np.array(train_x, dtype=float)
        train_y = np.array(train_y, dtype=float)

        self.feature_num = train_x.shape[0]
        self.total_num = train_x.shape[1]

        # 把y合并到x尾部
        train_input = np.concatenate(
            (train_x, train_y.reshape(1, self.total_num))
        ).T

        # 节点栈
        node_stack = []

        # 加入第一个根节点
        self.root = KdTreeNode(depth=0)
        node_stack.append((self.root, train_input))

        while node_stack:
            # 出栈一个节点, 预算一下中位数下标
            node, train_input = node_stack.pop()
            # print(train_input)
            median_num = int(len(train_input)/2)

            # 按第(深度%特征数)个特征值排序, 取中位数给node
            train_input = train_input[
                train_input[:, node.depth % self.feature_num].argsort()
            ]
            node.vector = train_input[median_num][:-1]
            node.category = train_input[median_num][-1]

            # print(node.depth, node.vector)

            # 右子树
            if len(train_input)-1 - median_num > 0:
                right_node = KdTreeNode(depth=node.depth+1, pre_node=node)
                node.right_node = right_node
                node_stack.append((right_node, train_input[median_num+1:]))
            # 左子树
            if median_num > 0:
                left_node = KdTreeNode(depth=node.depth+1, pre_node=node)
                node.left_node = left_node
                node_stack.append((left_node, train_input[:median_num]))

        print('KdTree has created.')

    def search_tree(self, x, k=1, p=2):
        """搜索KdTree算法

        输入:
            x: 待判别的实例
            k: k近邻
            p: 距离公式范数
        输出:
            返回x在KdTree中对应的类别, 与train_y的取值有关
        """

        if self.root is None:
            print("No KdTree!")
            return

        x = np.array(x, dtype=float)
        k_neighbor = []
        node = self.root
        node_stack = [node]

        # 搜索到根节点为止
        while node_stack:
            
            while True:
                node = node_stack[-1]
                feature_cmp_num = node.depth % self.feature_num

                # 比较当前维的值大小分别进入左右子树, 直到没有子树为止
                if x[feature_cmp_num] < node.vector[feature_cmp_num]:
                    if node.left_node is not None:
                        node = node.left_node
                        node_stack.append(node)
                    else:
                        break
                else:
                    if node.right_node is not None:
                        node = node.right_node
                        node_stack.append(node)
                    else:
                        break
            
            # 如果根节点被弹出就停止回溯
            while node_stack:
                node = node_stack.pop()
                feature_cmp_num = node.depth % self.feature_num

                # 点集没有k个的时候直接添加
                # 有k个时候更新最远的点
                if len(k_neighbor) < k:
                    k_neighbor.append(
                        (np.linalg.norm(x-node.vector, ord=p), node.category)
                    )
                else:
                    k_neighbor.sort()
                    if np.linalg.norm(x-node.vector, ord=p) < k_neighbor[-1][0]:
                        k_neighbor.pop()
                        k_neighbor.append(
                            (np.linalg.norm(x-node.vector, ord=p), node.category)
                        )

                # 根据点集里最远的距离和当前维的节点的垂直距离
                # 判断是否要进另外一个子树
                k_neighbor.sort()
                if k_neighbor[-1][0] > abs(x[feature_cmp_num]-node.vector[feature_cmp_num]):
                    # 当前维x小于节点的话要访问右子树
                    if x[feature_cmp_num] < node.vector[feature_cmp_num] and node.right_node is not None:
                        node_stack.append(node.right_node)
                        break
                    elif x[feature_cmp_num] >= node.vector[feature_cmp_num] and node.left_node is not None:
                        node_stack.append(node.left_node)
                        break
                    else:
                        # 不要进子树就继续回溯
                        continue
        
        # 计算各类数量
        neighbor_count = {}
        for neighbor in k_neighbor:
            if neighbor_count.get(neighbor[1]) is None:
                neighbor_count[neighbor[1]] = 1
            else:
                neighbor_count[neighbor[1]] += 1

        # 表决
        result = None
        max_count = 0
        for i in neighbor_count.items():
            if i[1] > max_count:
                result = i[0]
                max_count = i[1]
        return result
