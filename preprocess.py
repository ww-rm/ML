import json
import os
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english', True)
lemmatizer = WordNetLemmatizer()


def loadRawText(filepath):
    data = {
        'categories': [],
        'counts': [],
        'texts': []
    }

    categories = os.listdir(filepath)
    for category in categories:
        text_ids = os.listdir(filepath+'/'+category)
        data['categories'].append(category)
        data['counts'].append(len(text_ids))
        for text_id in text_ids:
            with open(filepath+'/'+category+'/'+text_id, encoding='utf8', errors='ignore') as text:
                data['texts'].append(text.read())

    return data


def sparse2dense(sparse, length):
    """length就是总词数

    sparse格式: [
        [[项号, 值], ...], ...
    ]
    """
    dense = []

    for each_sparse in sparse:
        dense_row = [0]*length
        for index, value in each_sparse:
            dense_row[index] = value
        dense.append(dense_row)

    return dense


def dense2sparse(dense):
    """
    dense格式: [
        [值, ...], ...
    ]
    """

    sparse = []

    for each_row in dense:
        each_sparse = []
        for index, value in enumerate(each_row):
            # 保存非0值
            if abs(value) > 1e-6:
                each_sparse.append([index, value])
        sparse.append(each_sparse)

    return sparse


def myTokenizer(text):
    tokens = re.findall(r'[a-zA-Z0-9]{3,15}', text)
    tokens = [lemmatizer.lemmatize(each) for each in tokens]
    tokens = [stemmer.stem(each) for each in tokens]

    return tokens


def preProcess(data, countpath, tfidfpath, tokenizer=None, wordsdict=None):
    """
    把读取的data向量化, 并保存稀疏矩阵到filepath
    """

    # 词频矩阵
    print('转换词频...')
    vectorizer = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer, vocabulary=wordsdict)
    vectors = vectorizer.fit_transform(data.get('texts', ''))
    vectors_array = vectors.toarray()

    # 保存词典
    if wordsdict is None:
        print('保存词典...')
        wordsdict = vectorizer.get_feature_names()  # {elem: index for index, elem in enumerate(vectorizer.get_feature_names())}
        with open('./data/words.dict', 'w', encoding='utf8') as f:
            json.dump(wordsdict, f)

    # 保存词频和tfidf的稀疏矩阵
    print('保存词频稀疏矩阵...')
    cur_index = 0
    for category, category_count in zip(data.get('categories'), data.get('counts')):
        with open(countpath+'/'+category, 'w', encoding='utf8') as f:
            tmp = dense2sparse(vectors_array[cur_index: cur_index+category_count].tolist())
            json.dump(tmp, f)
        cur_index += category_count
        del tmp

    # 转换成tfidf矩阵
    print('计算tfidf...')
    tfidf_trans = TfidfTransformer()
    tfidf_vectors = tfidf_trans.fit_transform(vectors)
    del vectors_array, vectors, vectorizer
    tfidf_vectors_array = tfidf_vectors.toarray()


    print('保存tfidf稀疏矩阵...')
    cur_index = 0
    for category, category_count in zip(data.get('categories'), data.get('counts')):
        with open(tfidfpath+'/'+category, 'w', encoding='utf8') as f:
            tmp = dense2sparse(tfidf_vectors_array[cur_index: cur_index+category_count].tolist())
            json.dump(tmp, f)
        cur_index += category_count
        del tmp

    print('预处理结束...')


if __name__ == "__main__":
    pass
    data = loadRawText('./data/20news-bydate-train')
    print('catagroies:', data.get('categories'))
    print('counts', data.get('counts'))
    preProcess(data, './data/sparse-train', './data/tfidf-sparse-train', myTokenizer)

    with open('./data/words.dict', encoding='utf8') as f:
        wordsdict = json.load(f)
    data = loadRawText('./data/20news-bydate-test')
    print('catagroies:', data.get('categories'))
    print('counts', data.get('counts'))
    preProcess(data, './data/sparse-test', './data/tfidf-sparse-test', myTokenizer, wordsdict=wordsdict)
