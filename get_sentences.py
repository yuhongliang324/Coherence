__author__ = 'yuhongliang324'
import re, os
from utils import accident_train_root, accident_test_root, earthquake_train_root, earthquake_test_root, dn
import cPickle

preprocessed_root = os.path.join(dn, 'preprocessed')
accident_train_sents_pkl = os.path.join(preprocessed_root, 'accident_train.pkl')
accident_test_sents_pkl = os.path.join(preprocessed_root, 'accident_test.pkl')
earthquake_train_sents_pkl = os.path.join(preprocessed_root, 'earthquake_train.pkl')
earthquake_test_sents_pkl = os.path.join(preprocessed_root, 'earthquake_test.pkl')

pattern = re.compile(r'\(\w+? [A-Z]+? "(.+?)"\)')


def get_all_sentences(root_path, out_pkl):
    files = os.listdir(root_path)
    files.sort()
    doc_sents_list = {}
    for fn in files:
        if not fn.endswith('.parsed'):
            continue
        sp = fn.split('.')
        doc = '.'.join(sp[:2])
        if doc not in doc_sents_list:
            doc_sents_list[doc] = []
        fpath = os.path.join(root_path, fn)
        sents = get_sents(fpath)
        if sents is None:
            continue
        doc_sents_list[doc].append(sents)
    f = open(out_pkl, 'wb')
    cPickle.dump(doc_sents_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    print 'Dumped to ' + out_pkl

    return doc_sents_list


def get_sents(parsed_file):
    reader = open(parsed_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    if len(lines) == 0:
        return None
    lines = lines[1:-1]

    num_sent = len(lines)
    sents = []
    for i in xrange(num_sent):
        words = re.findall(pattern, lines[i])
        words = map(lambda x: x.lower(), words)
        sent = ' '.join(words) + '.'
        sents.append(sent)
    return sents


if __name__ == '__main__':
    get_all_sentences(accident_train_root, accident_train_sents_pkl)
    get_all_sentences(accident_test_root, accident_test_sents_pkl)
    get_all_sentences(earthquake_train_root, earthquake_train_sents_pkl)
    get_all_sentences(earthquake_test_root, earthquake_test_sents_pkl)

