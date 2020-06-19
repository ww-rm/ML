项目目录结构:

./: 根目录下有所有的源文件以及包的需求
    preprocess.py: 预处理过程源文件
    bayes.py: 贝叶斯模型源文件
    bayes_test.py: 贝叶斯模型测试与调参文件
    perceptron.py: 感知机模型源文件
    perceptron_test.py: 感知机模型测试与调参文件
    test.py: 一键测试文件, 从数据的预处理到两个模型的测试结果

./data/
    20news-bydate-train/: 原始训练集
    20news-bydate-test/: 原始测试集
    pre-train/: 预处理之后的训练集, 包括稀疏矩阵和标签值
    pre-test/: 预处理之后的测试集, 内容同上
    stopwords.dict: 预处理中的停用词词典
    wordsdict.dict: 预处理之后从测试集中生成的词典

./model/: 存储各分类器训练之后的模型文件
