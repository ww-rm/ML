import json
import os
import re

import scipy.sparse
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



def myTokenizer(text):
    stemmer = SnowballStemmer('english', True)
    lemmatizer = WordNetLemmatizer()
    tokens = re.findall(r'[a-zA-Z0-9]{3,15}', text)
    tokens = [lemmatizer.lemmatize(each) for each in tokens]
    tokens = [stemmer.stem(each) for each in tokens]

    return tokens


def preProcess(datapath, savepath, categories=None, tokenizer=None, wordsdict=None):
    """
    从datapath读取原始数据, 并保存稀疏矩阵到savepath
    """

    stop_words = stopwords.words('english')

    # 读取数据
    print('read data...')
    data = datasets.load_files(datapath, categories=categories, encoding='utf8', decode_error='ignore')

    # 词频矩阵
    print('calculate words count...')
    vectorizer = CountVectorizer(stop_words=stop_words, tokenizer=tokenizer, vocabulary=wordsdict)
    vectors = vectorizer.fit_transform(data.get('data', ''))

    # 保存词典
    if wordsdict is None:
        print('save wordsdict...')
        wordsdict = vectorizer.get_feature_names()
        with open('./data/wordsdict.dict', 'w', encoding='utf8') as f:
            json.dump(wordsdict, f)

    # 转换成tfidf矩阵
    print('calculate tfidf...')
    tfidf_trans = TfidfTransformer()
    tfidf_vectors = tfidf_trans.fit_transform(vectors)
    
    # 保存词频和tfidf的稀疏矩阵
    print('save words count sparse...')
    scipy.sparse.save_npz(savepath+'/'+'count.npz', vectors.tocoo())

    print('save tfidf sparse...')
    scipy.sparse.save_npz(savepath+'/'+'tfidf.npz', tfidf_vectors.tocoo())

    print('save labels...')
    with open(savepath+'/'+'labels.json', 'w', encoding='utf8') as f:
        json.dump(data.get('target').tolist(), f)



if __name__ == "__main__":
    categories = os.listdir('./data/20news-bydate-train')
    # 训练集
    preProcess('./data/20news-bydate-train', './data/pre-train', categories, myTokenizer)
    # 测试集
    with open('./data/wordsdict.dict', encoding='utf8') as f:
        wordsdict = json.load(f)
    preProcess('./data/20news-bydate-test/', './data/pre-test', categories, myTokenizer, wordsdict=wordsdict)
