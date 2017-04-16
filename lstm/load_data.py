__author__ = 'yuhongliang324'
import numpy
import theano
import cPickle
import os
from utils import dn

preprocessed_root = os.path.join(dn, 'preprocessed')
accident_train_vecs_pkl = os.path.join(preprocessed_root, 'accident_vecs_train.pkl')
accident_test_vecs_pkl = os.path.join(preprocessed_root, 'accident_vecs_test.pkl')
earthquake_train_vecs_pkl = os.path.join(preprocessed_root, 'earthquake_vecs_train.pkl')
earthquake_test_vecs_pkl = os.path.join(preprocessed_root, 'earthquake_vecs_test.pkl')


def load(dataset='a'):
    if dataset.startswith('a'):
        train_pkl = accident_train_vecs_pkl
        test_pkl = accident_test_vecs_pkl
    else:
        train_pkl = earthquake_train_vecs_pkl
        test_pkl = earthquake_test_vecs_pkl

    print train_pkl
    reader = open(train_pkl)
    doc_vecs_train_list = cPickle.load(reader)
    reader.close()
    reader = open(test_pkl)
    doc_vecs_test_list = cPickle.load(reader)
    reader.close()

    Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train = load_set(doc_vecs_train_list)
    Xs_test, y_test, start_batches_test, end_batches_test, len_batches_test = load_set(doc_vecs_test_list)

    return Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train,\
           Xs_test, y_test, start_batches_test, end_batches_test, len_batches_test


def load_set(doc_vecs_list):
    maxlen = 0
    n = 0
    dim = doc_vecs_list.values()[0][0].shape[-1]
    for vecs_list in doc_vecs_list.values():
        n += len(vecs_list)
        for vecs in vecs_list:
            if vecs.shape[0] > maxlen:
                maxlen = vecs.shape[0]
    Xs = []
    y = []
    start_batches, end_batches, len_batches = [], [], []
    # docs = doc_vecs_list.keys()  if the document names are needed

    cur = 0
    for doc, vecs_list in doc_vecs_list.items():
        size = len(vecs_list)
        vecs = vecs_list[0]
        length = vecs.shape[0]
        start_batches.append(cur)
        end_batches.append(cur + size)
        len_batches.append(length)

        for i in xrange(size):
            length = vecs_list[i].shape[0]
            X = numpy.zeros((maxlen, dim))
            X[:length, :] = vecs_list[i]
            Xs.append(X)
            if i == 0:
                y.append(1)
            else:
                y.append(0)
        cur += size

    Xs = numpy.stack(Xs, axis=0).astype(theano.config.floatX)
    y = numpy.asarray(y, dtype=theano.config.floatX)

    return Xs, y, start_batches, end_batches, len_batches


def test1():
    load('a')


if __name__ == '__main__':
    test1()
