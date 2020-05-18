import json
import os
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import scipy.io
import scipy.sparse
from sklearn import datasets


def myTokenizer(text):
    stemmer = SnowballStemmer('english', True)
    lemmatizer = WordNetLemmatizer()
    tokens = re.findall(r'[a-zA-Z0-9]{3,15}', text)
    tokens = [lemmatizer.lemmatize(each) for each in tokens]
    tokens = [stemmer.stem(each) for each in tokens]

    return tokens


def preProcess(datapath, savepath, tokenizer=None, wordsdict=None):
    """
    把读取的data向量化, 并保存稀疏矩阵到filepath
    """

    stop_words = stopwords.words('english')
    with open('./data/categories.json', encoding='utf8') as f:
        categories = json.load(f)

    print('读取文件...')
    data = datasets.load_files(datapath, categories=categories, encoding='utf8', decode_error='ignore')
    # print(data.get('target_names'))
    # 词频矩阵
    print('计算词频...')
    vectorizer = CountVectorizer(stop_words=stop_words, tokenizer=tokenizer, vocabulary=wordsdict)
    vectors = vectorizer.fit_transform(data.get('data', ''))

    # 保存词典
    if wordsdict is None:
        print('保存词典...')
        wordsdict = vectorizer.get_feature_names()
        with open('./data/wordsdict.dict', 'w', encoding='utf8') as f:
            json.dump(wordsdict, f)

    # 保存词频和tfidf的稀疏矩阵
    print('保存词频稀疏矩阵...')
    scipy.io.mmwrite(savepath+'/'+'count.mtx', vectors.tocoo())

    # 转换成tfidf矩阵
    print('计算tfidf...')
    tfidf_trans = TfidfTransformer()
    tfidf_vectors = tfidf_trans.fit_transform(vectors)

    print('保存tfidf稀疏矩阵...')
    scipy.io.mmwrite(savepath+'/'+'tfidf.mtx', tfidf_vectors.tocoo())

    print('保存标签值...')
    with open(savepath+'/'+'labels.json', 'w', encoding='utf8') as f:
        json.dump(data.get('target').tolist(), f)

    print('预处理结束...')


if __name__ == "__main__":
    pass
    # 保存类别列表
    # categories = os.listdir('./data/20news-bydate-train')
    # with open('./data/categories.json', 'w', encoding='utf8') as f:
    #     json.dump(categories, f)
    # # 训练集
    # preProcess('./data/20news-bydate-train', './data/pre-train', myTokenizer)
    # # 测试集
    # with open('./data/wordsdict.dict', encoding='utf8') as f:
    #     wordsdict = json.load(f)
    # preProcess('./data/20news-bydate-test/', './data/pre-test', myTokenizer, wordsdict=wordsdict)
