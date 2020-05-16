# 这个文件是用来测试sklearn的贝叶斯文本分类结果的, 用来对比f1得分的

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

train_data = datasets.load_files('./data/20news-bydate-train', encoding='utf8', decode_error='ignore')
test_data = datasets.load_files('./data/20news-bydate-test', encoding='utf8', decode_error='ignore')
print('转换词频...')
# print(train_data.get('data'))
vectorizer1 = CountVectorizer()
vectors1 = vectorizer1.fit_transform(train_data.get('data'))
vectorizer2 = CountVectorizer(vocabulary=vectorizer1.get_feature_names())
vectors2 = vectorizer2.fit_transform(test_data.get('data'))
# print(vectors2.toarray().shape)
NB = MultinomialNB(0.071)
NB.fit(vectors1, train_data.get('target'))
score = NB.predict(vectors2)
a = classification_report(test_data.get('target'), score)
print(a)