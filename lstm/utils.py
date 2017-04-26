__author__ = 'yuhongliang324'
import os
from get_sentences import accident_train_sents_root, accident_test_sents_root,\
    earthquake_train_sents_root, earthquake_test_sents_root, preprocessed_root
import theano
import numpy
import cPickle

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