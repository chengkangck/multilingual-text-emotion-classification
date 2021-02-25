import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def load_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_tensor(train_set)
    valid_set_x, valid_set_y = make_tensor(valid_set)
    test_set_x, test_set_y = make_tensor(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def classify(data):

    df = pd.read_csv(data)
    y = df.iloc[:, 1]
    x = df.iloc[:, 2:]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
    clf = svm.SVC(C=1, probability=True)
    clf.fit(X_train, y_train)
    return y_train,y_test , clf.predict(X_test)


def load_pickle(f):

    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret
