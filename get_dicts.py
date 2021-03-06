__author__ = 'yuhongliang324'
import re, os
from utils import accident_train_root, accident_test_root, earthquake_train_root, earthquake_test_root, dn
import cPickle
import numpy as np

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


def create_dicts():
    # Create two dicts: one id to vec matrix called id_vec, one token to id dictionary called token_id.
    # where id is the token id we manually assigned to each token, vec is the token's embedding.

    dict_pkl = os.path.join(preprocessed_root, 'dict.pkl')
    with open(dict_pkl, 'r') as f:
        token_emb_dict = cPickle.load(f)

    print len(token_emb_dict.keys())

    token_file_root_paths = [accident_train_sents_root, accident_test_sents_root, earthquake_train_sents_root, earthquake_test_sents_root]

    # vocab_size = 6086
    # vocab_size = 5957
    vocab_size = 6021
    embedding_dim = 300
    id_vec = np.zeros((vocab_size + 1, embedding_dim))
    token_id = {}
    cur_id = 1
    count_oov = 0
    for path in token_file_root_paths:
        files = os.listdir(path)
        for fn in files:
            with open(os.path.join(path, fn), 'r') as f:
                for raw_line in f:
                    tokens = raw_line.rstrip().split()

                    for t in tokens:
                        if t in token_id:
                            continue
                        else:
                            if t in token_emb_dict:
                                token_id[t] = cur_id
                                id_vec[cur_id, :] = token_emb_dict[t]
                                cur_id += 1
                            else:
                                token_id[t] = 0 # OOV, UNKNOWN
                                print 'token %s not shown in the dict.pkl' % t
                                count_oov += 1
        print 'cur_id: %d' % cur_id
    oov_emb = id_vec.sum(axis=0) / vocab_size
    id_vec[0, :] = oov_emb

    print 'count_oov: %d ' % count_oov

    np.save(os.path.join(preprocessed_root, 'id_vec_matrix.npy'), id_vec)

    with open(os.path.join(preprocessed_root, 'token_id_dict.pkl'), 'wb') as f:
        cPickle.dump(token_id, f, protocol=cPickle.HIGHEST_PROTOCOL)


def create_pid2pos_pos2pid_dict():
    pos_file_root_paths = [accident_train_sents_pos_root, accident_test_sents_pos_root, earthquake_train_sents_pos_root, earthquake_test_sents_pos_root]

    pid_pos = {}
    pos_pid = {}
    cur_id = 0
    for path in pos_file_root_paths:
        files = os.listdir(path)
        for fn in files:
            with open(os.path.join(path, fn), 'r') as f:
                for raw_line in f:
                    pos_tags = raw_line.rstrip().split()

                    for p in pos_tags:
                        if p in pos_pid:
                            continue
                        else:
                            pos_pid[p] = cur_id
                            pid_pos[cur_id] = p
                            cur_id += 1

        print 'cur_id: %d' % cur_id

    with open(os.path.join(preprocessed_root, 'pid2pos_pos2pid_dict.pkl'), 'wb') as f:
        cPickle.dump([pid_pos, pos_pid], f, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # test3()
    # create_dicts()
    create_pid2pos_pos2pid_dict()
