__author__ = 'yuhongliang324'
import os
from scipy.io import loadmat
import numpy
import random
import theano
from sklearn.preprocessing import normalize
import cPickle
from get_sentvecs import accident_train_vecs_pkl, accident_test_vecs_pkl,\
    earthquake_train_vecs_pkl, earthquake_test_vecs_pkl


def load(dataset='a'):
    if dataset.startswith('a'):
        train_pkl = accident_train_vecs_pkl
        test_pkl = accident_test_vecs_pkl
    else:
        train_pkl = earthquake_train_vecs_pkl
        test_pkl = earthquake_test_vecs_pkl

    reader = open(train_pkl)
    doc_vecs_train_list = cPickle.load(train_pkl)
    reader.close()
    reader = open(test_pkl)
    doc_vecs_test_list = cPickle.load(test_pkl)
    reader.close()

    Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train = load_set(doc_vecs_train_list)


def load_set(doc_vecs_list):
    maxlen = 0
    n = 0
    dim = doc_vecs_list.values()[0][0].shape[-1]
    for vecs_list in doc_vecs_list.values():
        n += len(vecs_list)
        for vecs in vecs_list:
            if vecs.shape[0] > maxlen:
                maxlen = vecs.shape[0]
    Xs = numpy.zeros((n, maxlen, dim))
    y = []
    start_batches, end_batches, len_batches = [], [], []
    docs = doc_vecs_list.keys()

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

        cur += size

    return 1


def test1():
    load('a')


if __name__ == '__main__':
    test1()
