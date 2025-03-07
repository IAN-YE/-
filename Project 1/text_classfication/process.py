import read
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

def process():
    train_data = read.read_train_data()
    test_data = read.read_test_data()
    #tfidf_vec = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_vec = CountVectorizer(analyzer='word', stop_words='english')
    x_train = tfidf_vec.fit_transform(train_data['raw'])
    y_train = train_data['label'].values * 1
    x_test = tfidf_vec.transform(test_data['text'])

    return x_train.todense(), y_train, x_test.todense()

import numpy as np
x_train, y_train, x_test = process()
print(np.min(np.sum(x_train, axis=0)))
print(x_train.shape)