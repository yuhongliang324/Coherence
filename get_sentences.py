__author__ = 'yuhongliang324'
import re, os
from utils import accident_train_root, accident_test_root, earthquake_train_root, earthquake_test_root, dn
import cPickle

preprocessed_root = os.path.join(dn, 'preprocessed')
accident_train_sents_pkl = os.path.join(preprocessed_root, 'accident_train.pkl')
accident_test_sents_pkl = os.path.join(preprocessed_root, 'accident_test.pkl')
earthquake_train_sents_pkl = os.path.join(preprocessed_root, 'earthquake_train.pkl')
earthquake_test_sents_pkl = os.path.join(preprocessed_root, 'earthquake_test.pkl')

accident_train_sents_root = os.path.join(preprocessed_root, 'accident_train')
accident_test_sents_root = os.path.join(preprocessed_root, 'accident_test')
earthquake_train_sents_root = os.path.join(preprocessed_root, 'earthquake_train')
earthquake_test_sents_root = os.path.join(preprocessed_root, 'earthquake_test')

accident_train_sents_pos_root = os.path.join(preprocessed_root, 'accident_train_pos')
accident_test_sents_pos_root = os.path.join(preprocessed_root, 'accident_test_pos')
earthquake_train_sents_pos_root = os.path.join(preprocessed_root, 'earthquake_train_pos')
earthquake_test_sents_pos_root = os.path.join(preprocessed_root, 'earthquake_test_pos')

pattern = re.compile(r'\(\w+? ([A-Z]+?) ".+?"\)')


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


def get_sents2(parsed_file):
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


def write_all_sents(root_path, out_root, get_sents=get_sents):
    if not os.path.isdir(out_root):
        os.mkdir(out_root)
    files = os.listdir(root_path)
    files.sort()
    for fn in files:
        if not fn.endswith('.parsed'):
            continue
        sents = get_sents(os.path.join(root_path, fn))
        writer = open(os.path.join(out_root, fn).replace('parsed', 'txt'), 'w')
        if sents is None:
            writer.close()
            continue
        for sent in sents:
            writer.write(sent[:-1] + ' .\n')
        writer.close()


def test1():
    get_all_sentences(accident_train_root, accident_train_sents_pkl)
    get_all_sentences(accident_test_root, accident_test_sents_pkl)
    get_all_sentences(earthquake_train_root, earthquake_train_sents_pkl)
    get_all_sentences(earthquake_test_root, earthquake_test_sents_pkl)


def test2():
    write_all_sents(accident_train_root, accident_train_sents_root)
    write_all_sents(accident_test_root, accident_test_sents_root)
    write_all_sents(earthquake_train_root, earthquake_train_sents_root)
    write_all_sents(earthquake_test_root, earthquake_test_sents_root)


def test3():
    write_all_sents(accident_train_root, accident_train_sents_pos_root, get_sents=get_sents)
    write_all_sents(accident_test_root, accident_test_sents_pos_root, get_sents=get_sents)
    write_all_sents(earthquake_train_root, earthquake_train_sents_pos_root, get_sents=get_sents)
    write_all_sents(earthquake_test_root, earthquake_test_sents_pos_root, get_sents=get_sents)

if __name__ == '__main__':
    test3()

