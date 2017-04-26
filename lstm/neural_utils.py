__author__ = 'yuhongliang324'

import sys
sys.path.append('..')

import os
import theano
import numpy
import cPickle
from load_data import load_dict, accident_train_sents_pkl2, accident_test_sents_pkl2,\
    earthquake_train_sents_pkl2, earthquake_test_sents_pkl2

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


def get_sentIDs(root_path, token_ID, out_pkl):
    def get_sents(fn):
        reader = open(fn)
        sents = reader.readlines()
        reader.close()
        sents = map(lambda x: x.strip(), sents)
        return sents

    sentence_ID = {}
    sentences = []
    sent_tokenids = []
    curID = 0
    files = os.listdir(root_path)
    files.sort()
    doc_paras = {}
    for fn in files:
        if not fn.endswith('.txt'):
            continue
        fpath = os.path.join(root_path, fn)
        sents = get_sents(fpath)
        for sent in sents:
            if sent not in sentence_ID:
                sentence_ID[sent] = curID
                sentences.append(sent)
                tokens = sent.split()
                tokenids = [token_ID[token] for token in tokens]
                sent_tokenids.append(tokenids)
                curID += 1

    for fn in files:
        if not fn.endswith('.txt'):
            continue
        sp = fn.split('.')
        doc = '.'.join(sp[:2])

        fpath = os.path.join(root_path, fn)
        if doc not in doc_paras:
            doc_paras[doc] = []
        sents = get_sents(fpath)
        sent_ids = [sentence_ID[sent] for sent in sents]
        doc_paras[doc].append(sent_ids)

    print len(sentence_ID), len(sentences)
    f = open(out_pkl, 'wb')
    cPickle.dump([sentences, sent_tokenids, doc_paras], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def test1():
    tokens = get_dict()
    get_vectors(tokens)


def test11():
    from load_data import token_id_pkl
    reader = open(token_id_pkl)
    tokens = cPickle.load(reader).keys()
    reader.close()
    get_vectors(tokens)


def test2():
    token_id, _ = load_dict()
    get_sentIDs(accident_train_sents_root, token_id, accident_train_sents_pkl2)
    get_sentIDs(accident_test_sents_root, token_id, accident_test_sents_pkl2)
    get_sentIDs(earthquake_train_sents_root, token_id, earthquake_train_sents_pkl2)
    get_sentIDs(earthquake_test_sents_root, token_id, earthquake_test_sents_pkl2)


if __name__ == '__main__':
    test2()
