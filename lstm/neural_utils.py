__author__ = 'yuhongliang324'

import sys
sys.path.append('..')

import os
import theano
import numpy
import cPickle
from data_path import accident_train_root

preprocessed_root = '/usr0/home/hongliay/code/Coherence/preprocessed'
accident_train_sents_pkl = os.path.join(preprocessed_root, 'accident_train.pkl')
accident_test_sents_pkl = os.path.join(preprocessed_root, 'accident_test.pkl')
earthquake_train_sents_pkl = os.path.join(preprocessed_root, 'earthquake_train.pkl')
earthquake_test_sents_pkl = os.path.join(preprocessed_root, 'earthquake_test.pkl')

accident_train_sents_root = os.path.join(preprocessed_root, 'accident_train')
accident_test_sents_root = os.path.join(preprocessed_root, 'accident_test')
earthquake_train_sents_root = os.path.join(preprocessed_root, 'earthquake_train')
earthquake_test_sents_root = os.path.join(preprocessed_root, 'earthquake_test')

wordvec_file = '/usr0/home/hongliay/word_vectors/glove.840B.300d.txt'
dict_pkl = os.path.join(preprocessed_root, 'dict.pkl')


def get_dict():
    tokens = set()

    def get_dict2(fn):
        reader = open(fn)
        lines = reader.readlines()
        reader.close()
        lines = map(lambda x: x.strip(), lines)
        for line in lines:
            words = line.split()
            num_words = len(words)
            for i in xrange(1, num_words):
                if words[i] == '-lrb-':
                    words[i] = '('
                elif words[i] == '-rrb-':
                    words[i] = ')'
                tokens.add(words[i])
                if '-' in words[i]:
                    for w in words[i].split('-'):
                        tokens.add(w)

    def get_dict1(root_path):
        files = os.listdir(root_path)
        files.sort()
        for fn in files:
            if not fn.endswith('.txt'):
                continue
            get_dict2(os.path.join(root_path, fn))

    get_dict1(accident_train_sents_root)
    get_dict1(accident_test_sents_root)
    get_dict1(earthquake_train_sents_root)
    get_dict1(earthquake_test_sents_root)

    return tokens


def get_vectors(tokens, vec_file=wordvec_file, out_file=dict_pkl):
    token_vec = {}
    reader = open(vec_file)
    count = 0
    while True:
        line = reader.readline()
        if line:
            count += 1
            if count % 100000 == 0:
                print count
            line = line.strip()
            sp = line.split()
            if sp[0] not in tokens:
                continue
            tok = sp[0]
            vec = [float(x) for x in sp[1:]]
            vec = numpy.asarray(vec, dtype=theano.config.floatX)
            token_vec[tok] = vec
        else:
            break
    reader.close()
    f = open(out_file, 'wb')
    cPickle.dump(token_vec, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    oov = []
    for token in tokens:
        if token not in token_vec:
            oov.append(token)
    print oov
    print len(oov), len(token_vec)
    return token_vec


def get_sentIDs(root_path, out_pkl):
    def get_sents(fn):
        reader = open(fn)
        sents = reader.readlines()
        reader.close()
        sents = map(lambda x: x.strip(), sents)
        return sents

    sentences = []
    files = os.listdir(root_path)
    files.sort()
    doc_paras = {}
    startID, endID = 0, 0
    for fn in files:
        if not fn.endswith('.txt'):
            continue
        sp = fn.split('.')
        doc = '.'.join(sp[:2])
        fpath = os.path.join(root_path, fn)
        if doc not in doc_paras:
            doc_paras[doc] = []
            sents = get_sents(fpath)
            sentences += sents
            startID = endID
            endID += len(sents)
        else:
            sents = get_sents(fpath)
        sent_ids = []
        for sent in sents:
            found = False
            for i in xrange(startID, endID):
                if sent == sentences[i]:
                    sent_ids.append(i)
                    found = True
                    break
            if not found:
                print sent
        doc_paras[doc].append(sent_ids)
    print len(sentences)
    f = open(out_pkl, 'wb')
    cPickle.dump([sentences, doc_paras], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def test1():
    tokens = get_dict()
    get_vectors(tokens)


def test2():
    get_sentIDs(accident_train_root, 'tmp.pkl')


if __name__ == '__main__':
    test2()
