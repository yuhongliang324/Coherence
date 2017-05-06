__author__ = 'yuhongliang324'

import cPickle, os, numpy, theano
from theano import tensor as T
from gen import RNN
import argparse

preprocessed_root = '../preprocessed'
accident_train_pkl = os.path.join(preprocessed_root, 'accident_train2_final.pkl')
accident_test_pkl = os.path.join(preprocessed_root, 'accident_test2_final.pkl')
earthquake_train_pkl = os.path.join(preprocessed_root, 'earthquake_train2_final.pkl')
earthquake_test_pkl = os.path.join(preprocessed_root, 'earthquake_test2_final.pkl')


def load(pkl_file):
    reader = open(pkl_file)
    sents, E, xs, ys, discs, disc_labels = cPickle.load(reader)
    reader.close()
    lenxs, lenys = [], []
    maxlen = xs.shape[1]
    for x in xs:
        if x[-1] != -1:
            lenx = maxlen
        else:
            lenx = 0
            while x[lenx] != -1:
                lenx += 1
        lenxs.append(lenx)
        lenys.append(lenx + 1)
    lenxs = numpy.asarray(lenxs, dtype='int32')
    lenys = numpy.asarray(lenys, dtype='int32')
    return sents, E, xs, ys, lenxs, lenys, discs, disc_labels


def classify(train_pkl, test_pkl, attention=False, hidden_dim=128, drop=0., num_epoch=20):
    sents_train, E_old, xs_train, ys_train, lenxs_train, lenys_train, discs_train, disc_labels_train = load(train_pkl)
    sents_test, _, xs_test, ys_test, lenxs_test, lenys_test, discs_test, disc_labels_test = load(test_pkl)

    xs_train = theano.shared(xs_train, borrow=True)
    ys_train = theano.shared(ys_train, borrow=True)
    lenxs_train = theano.shared(lenxs_train, borrow=True)
    lenys_train = theano.shared(lenys_train, borrow=True)
    xs_test = theano.shared(xs_test, borrow=True)
    ys_test = theano.shared(ys_test, borrow=True)
    lenxs_test = theano.shared(lenxs_test, borrow=True)
    lenys_test = theano.shared(lenys_test, borrow=True)

    E = numpy.zeros((E_old.shape[0] + 1, E_old.shape[1]), dtype=theano.config.floatX)
    E[: -1] = E_old

    n_class, input_dim = E_old.shape[0], E_old.shape[1]
    model = RNN(E, input_dim, hidden_dim, n_class, attention=attention, drop=drop)
    if attention:
        variables = model.build_model_att()
    else:
        variables = model.build_model()

    x_full, y_full, lenx, leny, is_train, prob = variables['x'], variables['y'], variables['lenx'], variables['leny'],\
                                                 variables['is_train'], variables['prob']
    att, pred, loss, cost, updates = variables['att'], variables['pred'], variables['loss'], variables['cost'],\
                                     variables['updates']
    acc = variables['acc']
    Ctx = variables['ctx']

    xid, yid = T.iscalar(), T.iscalar()
    print 'Compiling function'
    train_model = theano.function(inputs=[xid, yid, is_train],
                                  outputs=[prob, acc, cost, Ctx], updates=updates,
                                  givens={
                                      x_full: xs_train[xid], y_full: ys_train[yid],
                                      lenx: lenxs_train[xid], leny: lenys_train[yid]
                                  },
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    test_model = theano.function(inputs=[xid, yid, is_train],
                                 outputs=[prob, acc, cost],
                                 givens={
                                     x_full: xs_test[xid], y_full: ys_test[yid],
                                     lenx: lenxs_test[xid], leny: lenys_test[yid]
                                 },
                                 on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'

    report_iter = 100

    for epoch_index in xrange(num_epoch):
        iter_index = 0
        prob_report, acc_report, cost_report = 0., 0., 0.
        prob_epoch, acc_epoch, cost_epoch = 0., 0., 0.
        for disc, label in zip(discs_train, disc_labels_train):
            if label == 0:
                continue
            n_sent = len(disc)
            for i in xrange(n_sent - 1):
                xid, yid = disc[i], disc[i + 1]

                prob, acc, cost, Ctx = train_model(xid, yid, 1)
                prob_report += prob
                acc_report += acc
                cost_report += cost

                prob_epoch += prob
                acc_epoch += acc
                cost_epoch += cost
                print lenys_train.get_value()[yid], Ctx.shape

                iter_index += 1
                if iter_index % report_iter == 0:
                    prob_report /= report_iter
                    acc_report /= report_iter
                    cost_report /= report_iter
                    print '\tEpoch = %d, iter = %d, prob = %.5f, acc = %.5f, cost = %.4f'\
                          % (epoch_index + 1, iter_index, prob_report, acc_report, cost_report)
                    prob_report, acc_report, cost_report = 0., 0., 0.
        prob_epoch /= iter_index
        acc_epoch /= iter_index
        cost_epoch /= iter_index
        print 'Training epoch = %d, prob = %.5f, acc = %.5f, cost = %.4f'\
              % (epoch_index + 1, prob_epoch, acc_epoch, cost_epoch)
        binary_classification(test_model, discs_test, disc_labels_test)


def binary_classification(test_model, discs_test, discs_labels_test):
    iter_index = 0
    prob_pred = []
    count = 0
    for disc, label in zip(discs_test, discs_labels_test):
        n_sent = len(disc)
        p = 0.
        for i in xrange(n_sent - 1):
            xid, yid = disc[i], disc[i + 1]
            prob, acc, cost = test_model(xid, yid, 0)
            p += prob
            iter_index += 1
        p /= n_sent - 1
        prob_pred.append(p)
        count += 1
        if count % 100 == 0:
            print count, '/', len(discs_test)

    acc = Acc_comp(discs_labels_test, prob_pred)

    print 'Test Accuracy = %.5f' % acc


def Acc_comp(y_actual, y_predicted):
    right = 0
    total = 0
    size = len(y_actual)
    for i in xrange(size):
        if y_actual[i] == 1:
            for j in xrange(i + 1, size):
                if y_actual[j] == 1:
                    break
                else:
                    if y_predicted[i] > y_predicted[j]:
                        right += 1.
                    total += 1.
    acc = right / total
    return acc


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-doc', type=str, default='a')
    parser.add_argument('-hid', type=int, default=128)
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-att', type=bool, default=False)
    args = parser.parse_args()
    print 'att', args.att
    if args.doc.startswith('a'):
        train_pkl = accident_train_pkl
        test_pkl = accident_test_pkl
    else:
        train_pkl = earthquake_train_pkl
        test_pkl = earthquake_test_pkl
    classify(train_pkl, test_pkl, attention=args.att, hidden_dim=args.hid, drop=args.drop)


if __name__ == '__main__':
    test1()
