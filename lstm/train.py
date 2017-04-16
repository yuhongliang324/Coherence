__author__ = 'yuhongliang324'

import argparse
import sys

sys.path.append('../')
import numpy
from optimize import train
from load_data import load
import theano


def experiment(dataset='a', update='adam2', lamb=0., drop=0., sq_loss=False, model='lstml', share=False, category=True,
               early_stop=False):

    Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train,\
    Xs_test, y_test, start_batches_test, end_batches_test, len_batches_test = load(dataset=dataset)

    if category:
        y_train = y_train.astype('int32')
    else:
        y_train = y_train.astype(theano.config.floatX)

    if category:
        cnt = [0, 0]
        for i in xrange(y_test.shape[0]):
            cnt[y_test[i]] += 1
        cnt = numpy.asarray(cnt)
        acc = numpy.max(cnt) / float(y_test.shape[0])
        cl = numpy.argmax(cnt)
        print 'Majority Accuracy = %f, Majority Class = %d' % (acc, cl)
    else:
        rating_mean = numpy.mean(y_train)
        mae = numpy.abs(y_test - rating_mean)
        mae = numpy.mean(mae)
        print 'MAE of Average Prediction = %f' % mae

    print Xs_train.shape, Xs_test.shape

    inputs_train = (Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train)
    inputs_test = (Xs_test, y_test, start_batches_test, end_batches_test, len_batches_test)

    hidden_dim = min(512, Xs_train.shape[-1])

    train(inputs_train, inputs_test, hidden_dim=hidden_dim, update=update, sq_loss=sq_loss, num_epoch=40,
          lamb=lamb, model=model, share=share, category=category, drop=drop, num_class=2, early_stop=early_stop)


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='a')
    parser.add_argument('-update', type=str, default='adam2')
    parser.add_argument('-lamb', type=float, default=0.)
    parser.add_argument('-model', type=str, default='lstml')
    parser.add_argument('-share', type=int, default=0)
    parser.add_argument('-cat', type=int, default=0)
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-sq', type=int, default=1)
    args = parser.parse_args()

    args.share = bool(args.share)
    args.cat = bool(args.cat)
    args.sq = bool(args.sq)

    if args.feat == 'text':
        es = True
    else:
        es = False
    experiment(dataset=args.data, update=args.update, lamb=args.lamb, drop=args.drop, sq_loss=args.sq,
               model=args.model, share=args.share, category=False, early_stop=es)


if __name__ == '__main__':
    test1()
