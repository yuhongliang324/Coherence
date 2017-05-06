__author__ = 'yuhongliang324'
import cPickle
import numpy


def load_train(pkl_file):
    reader = open(pkl_file)
    _, sent_tokenids, doc_paras = cPickle.load(reader)
    reader.close()

    X_list, Y_list = [], []
    start_batches, end_batches = [], []
    cur = 0
    maxlenX, maxlenY = 0, 0
    n_sent = 0
    for doc, paras in doc_paras.items():
        sent_ids = paras[0]
        size = len(sent_ids)
        start_batches.append(cur)
        end_batches.append(cur + size)
        cur += size
        for i in xrange(size - 1):
            pind, nind = sent_ids[i], sent_ids[i + 1]
            X_list.append(sent_tokenids[pind])
            Y_list.append([-1] + sent_tokenids[nind])
            maxlenX = max(maxlenX, len(sent_tokenids[pind]))
            maxlenY = max(maxlenY, len(sent_tokenids[nind]) + 1)
            n_sent += 1

    X = numpy.zeros((n_sent, maxlenX), dtype='int32') - 1
    Y = numpy.zeros((n_sent, maxlenY), dtype='int32') - 1
    lenX, lenY = [], []
    for i in xrange(n_sent):
        x = numpy.asarray(X_list[i], dtype='int32')
        l = x.shape[0]
        X[i, -l:] = x
        lenX.append(l)
        y = numpy.asarray(Y_list[i], dtype='int32')
        l = y.shape[0]
        Y[i, :l] = y
        lenY.append(l)

    lenX = numpy.asarray(lenX, dtype='int32')
    lenY = numpy.asarray(lenY, dtype='int32')

    return X, Y, lenX, lenY, start_batches, end_batches

