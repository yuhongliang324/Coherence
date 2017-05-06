__author__ = 'dfan'

from utils import dn

import os
import cPickle
import numpy as np

preprocessed_root = os.path.join(dn, 'preprocessed')
accident_train_sents_pkl3 = os.path.join(preprocessed_root, 'accident_train3.pkl')
accident_test_sents_pkl3 = os.path.join(preprocessed_root, 'accident_test3.pkl')
earthquake_train_sents_pkl3 = os.path.join(preprocessed_root, 'earthquake_train3.pkl')
earthquake_test_sents_pkl3 = os.path.join(preprocessed_root, 'earthquake_test3.pkl')
id_vec_matrix_npy = os.path.join(preprocessed_root, 'id_vec_matrix.npy')
pid2pos_pos2pid_dict_pkl = os.path.join(preprocessed_root, 'pid2pos_pos2pid_dict.pkl')

id_vec_matrix = np.load(id_vec_matrix_npy)

def load_pid2pos_pos2pid_dict():
    with open(pid2pos_pos2pid_dict_pkl, 'rb') as f:
        pid_pos_list, pos_pid_dict = cPickle.load(f)
    return pid_pos_list, pos_pid_dict

pid_pos_list, pos_pid_dict = load_pid2pos_pos2pid_dict()

rid_role_list = ['S', 'O', 'X', 'B', 'V', '?']
role_rid_dict = {'S': 0, 'O': 1, 'X': '2', 'B': 3, 'V': 4, '?': 5}


def process_pkl(pkl_file):
    f = open(pkl_file, 'r')
    orig_sentences, sentences, discourse_dict, sent_pids, sent_rids = cPickle.load(f)
    f.close()

    A = orig_sentences # List of sentence strings. Each string is space splitable.
    B = id_vec_matrix # Matrix of word embeddings. Row "t" is an embedding for word "t".


    max_sent_len = 0
    for sent in sentences:
        cur_sent_len = len(sent)
        if cur_sent_len > max_sent_len:
            max_sent_len = cur_sent_len

    sentences_matrix = -np.ones((len(sentences), max_sent_len), dtype='int32')
    sentences_matrix_with_start = -np.ones((len(sentences), max_sent_len + 1), dtype='int32')
    pos_matrix = -np.ones((len(sent_pids), max_sent_len), dtype='int32')
    pos_matrix_with_start = -np.ones((len(sent_pids), max_sent_len + 1), dtype='int32')
    role_matrix = -np.ones((len(sent_rids), max_sent_len), dtype='int32')
    role_matrix_with_start = -np.ones((len(sent_rids), max_sent_len + 1), dtype='int32')

    for row in xrange(len(sentences)):
        sent = sentences[row]
        pids = sent_pids[row]
        rids = sent_rids[row]

        assert len(sent) == len(pids)
        assert len(sent) == len(rids)

        sentences_matrix[row, 0:len(sent)] = sent
        sentences_matrix_with_start[row, 1:len(sent) + 1] = sent

        pos_matrix[row, 0:len(pids)] = pids
        pos_matrix_with_start[row, 1:len(pids) + 1] = pids

        role_matrix[row, 0:len(rids)] = rids
        role_matrix_with_start[row, 1:len(rids) + 1] = rids

    C = sentences_matrix # -1 padded matrix. Row "t" is a vector of word ids padded with -1 for sentence "t".
    D = sentences_matrix_with_start # Padded another column of -1 before the first column of C.

    G = pid_pos_list  # A dictionary of pos id to pos.
    H = pos_matrix  # -1 padded matrix. Row "t" is a vector of pos ids padded with -1 for sentence "t".
    I = pos_matrix_with_start  # Padded another column of -1 before the first column of G.

    J = rid_role_list  # A dictionary of role id to role.
    K = role_matrix  # -1 padded matrix. Row "t" is a vector of role ids padded with -1 for sentence "t".
    L = role_matrix_with_start  # Padded another column of -1 before the first column of I.

    discourse_list = []
    discourse_list_label = []
    for docname, discourse_list_for_a_doc in discourse_dict.iteritems():
        discourse_list.extend(discourse_list_for_a_doc)

        temp = [0] * len(discourse_list_for_a_doc)
        temp[0] = 1
        discourse_list_label.extend(temp)

    E = discourse_list # Each row is a discourse that has a series of sentences. Each element is a sentence id.
    F = discourse_list_label # Indicate each row of E is a coherent discourse or not.


    file_name, ext = os.path.splitext(pkl_file)
    outfile = file_name + '_final' + ext
    with open(outfile, 'wb') as f:
        cPickle.dump((A, B, C, D, E, F, G, H, I, J, K, L), f, protocol=cPickle.HIGHEST_PROTOCOL)



def write_pkls():
    process_pkl(accident_train_sents_pkl3)
    process_pkl(accident_test_sents_pkl3)
    process_pkl(earthquake_train_sents_pkl3)
    process_pkl(earthquake_test_sents_pkl3)

if __name__ == '__main__':
    write_pkls()