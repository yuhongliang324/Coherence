__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from lstm.theano_utils import Adam, Adam2, RMSprop, SGD, dropout


class RNN(object):
    def __init__(self, E, input_dim, hidden_dim, n_class, n_roles=0, hidden_dim_role=50, lamb=0., update='adam2',
                 drop=0.2):
        self.E = theano.shared(E, borrow=True)
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.n_class = n_class
        self.lamb = lamb
        self.drop = drop
        self.update = update
        self.rng = numpy.random.RandomState(1234)
        theano_seed = numpy.random.randint(2 ** 30)
        self.theano_rng = RandomStreams(theano_seed)

        self.W_enc_i, self.b_enc_i, self.U_enc_i, \
        self.W_enc_f, self.b_enc_f, self.U_enc_f, \
        self.W_enc_o, self.b_enc_o, self.U_enc_o, \
        self.W_enc_c, self.b_enc_c, self.U_enc_c = self.create_lstm_para()
        self.theta = [self.W_enc_i, self.b_enc_i, self.U_enc_i,
                      self.W_enc_f, self.b_enc_f, self.U_enc_f,
                      self.W_enc_o, self.b_enc_o, self.U_enc_o,
                      self.W_enc_c, self.b_enc_c, self.U_enc_c]

        self.W_dec_i, self.b_dec_i, self.U_dec_i, \
        self.W_dec_f, self.b_dec_f, self.U_dec_f, \
        self.W_dec_o, self.b_dec_o, self.U_dec_o, \
        self.W_dec_c, self.b_dec_c, self.U_dec_c = self.create_lstm_para()
        self.theta += [self.W_dec_i, self.b_dec_i, self.U_dec_i,
                       self.W_dec_f, self.b_dec_f, self.U_dec_f,
                       self.W_dec_o, self.b_dec_o, self.U_dec_o,
                       self.W_dec_c, self.b_dec_c, self.U_dec_c]
        '''
        if n_roles > 0:
            self.R, _ = self.init_para(n_roles, hidden_dim_role)
            self.theta.append(self.R)'''
        self.W, self.b = self.init_para(self.hidden_dim, self.n_class)
        self.theta += [self.W, self.b]

        print 'lambda =', self.lamb, 'last =', '#class =', self.n_class, 'drop =', self.drop, 'update =', self.update

        if self.update == 'adam':
            self.optimize = Adam
        elif self.update == 'adam2':
            self.optimize = Adam2
        elif self.update == 'rmsprop':
            self.optimize = RMSprop
        else:
            self.optimize = SGD

    def create_lstm_para(self):
        W_i, b_i = self.init_para(self.input_dim, self.hidden_dim)
        U_i, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_f, b_f = self.init_para(self.input_dim, self.hidden_dim)
        U_f, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_o, b_o = self.init_para(self.input_dim, self.hidden_dim)
        U_o, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_c, b_c = self.init_para(self.input_dim, self.hidden_dim)
        U_c, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        return [W_i, b_i, U_i, W_f, b_f, U_f, W_o, b_o, U_o, W_c, b_c, U_c]

    def init_para(self, d1, d2):
        W_values = numpy.asarray(self.rng.uniform(
            low=-numpy.sqrt(6. / float(d1 + d2)), high=numpy.sqrt(6. / float(d1 + d2)), size=(d1, d2)),
            dtype=theano.config.floatX)
        W = theano.shared(value=W_values, borrow=True)
        b_values = numpy.zeros((d2,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

    def l2(self):
        l2 = self.lamb * T.sum([T.sum(p ** 2) for p in self.theta])
        return l2

    def encode_step(self, X_t, C_tm1, H_tm1):
        i_t = T.nnet.sigmoid(T.dot(X_t, self.W_enc_i) + T.dot(H_tm1, self.U_enc_i) + self.b_enc_i)
        f_t = T.nnet.sigmoid(T.dot(X_t, self.W_enc_f) + T.dot(H_tm1, self.U_enc_f) + self.b_enc_f)
        o_t = T.nnet.sigmoid(T.dot(X_t, self.W_enc_o) + T.dot(H_tm1, self.U_enc_o) + self.b_enc_o)
        C_t = T.tanh(T.dot(X_t, self.W_enc_c) + T.dot(H_tm1, self.U_enc_c) + self.b_enc_c)
        C_t = i_t * C_t + f_t * C_tm1
        H_t = o_t * T.tanh(C_t)
        return C_t, H_t

    def decode_step(self, Yc_t, C_tm1, H_tm1):
        i_t = T.nnet.sigmoid(T.dot(Yc_t, self.W_dec_i) + T.dot(H_tm1, self.U_dec_i) + self.b_dec_i)
        f_t = T.nnet.sigmoid(T.dot(Yc_t, self.W_dec_f) + T.dot(H_tm1, self.U_dec_f) + self.b_dec_f)
        o_t = T.nnet.sigmoid(T.dot(Yc_t, self.W_dec_o) + T.dot(H_tm1, self.U_dec_o) + self.b_dec_o)
        C_t = T.tanh(T.dot(Yc_t, self.W_dec_c) + T.dot(H_tm1, self.U_dec_c) + self.b_dec_c)
        C_t = i_t * C_t + f_t * C_tm1
        H_t = o_t * T.tanh(C_t)  # (hid_dim,)
        return C_t, H_t

    def build_model(self):
        x_full = T.ivector()  # (max_len,)
        y_full = T.ivector()  # (max_len,)
        lenx = T.iscalar()
        leny = T.iscalar()
        x = x_full[: lenx]  # (lenx,)
        y = y_full[: leny]  # (leny,)
        X = self.E[x]  # (lenx, 300)
        Yc = self.E[y[:-1]]  # (leny - 1, 300)
        yn = y[1:]  # (leny - 1,)

        [_, H_enc], _ = theano.scan(self.encode_step, sequences=X,
                                    outputs_info=[T.zeros((self.hidden_dim,),
                                                          dtype=theano.config.floatX),
                                                  T.zeros((self.hidden_dim,),
                                                          dtype=theano.config.floatX)])
        rep_enc = H_enc[-1]  # (hidden_dim)
        # H_dec: (leny - 1, hid_dim)
        [_, H_dec], _ = theano.scan(self.decode_step, sequences=Yc,
                                    outputs_info=[T.zeros((self.hidden_dim,),
                                                          dtype=theano.config.floatX),
                                                  rep_enc])

        is_train = T.iscalar('is_train')

        rep_dec = T.dot(H_dec, self.W) + self.b  # (leny - 1, n_class)
        rep_dec = dropout(rep_dec, is_train, drop_ratio=self.drop)  # (leny - 1, n_class)

        prob = T.nnet.softmax(rep_dec)  # (leny - 1, n_class)
        yn_onehot = T.extra_ops.to_one_hot(yn, self.n_class)  # (leny - 1, n_class)
        prob_trueval = T.sum(yn_onehot, axis=1)  # (leny - 1,)
        prob_trueval = T.mean(prob_trueval)
        pred = T.argmax(prob, axis=-1)

        acc = T.mean(T.eq(pred, yn))
        loss = T.mean(T.nnet.categorical_crossentropy(prob, yn))
        cost = loss + self.l2()

        updates = self.optimize(cost, self.theta)

        ret = {'x': x_full, 'y': y_full, 'lenx': lenx, 'leny': leny, 'is_train': is_train, 'prob': prob_trueval,
               'att': None, 'pred': pred, 'loss': loss, 'cost': cost, 'updates': updates,
               'acc': acc}
        return ret
