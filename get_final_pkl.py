__author__ = 'dfan'

from utils import accident_train_root, accident_test_root, earthquake_train_root, earthquake_test_root, dn

import os
import cPickle
import numpy as np

preprocessed_root = os.path.join(dn, 'preprocessed')
accident_train_sents_pkl2 = os.path.join(preprocessed_root, 'accident_train2.pkl')
accident_test_sents_pkl2 = os.path.join(preprocessed_root, 'accident_test2.pkl')
earthquake_train_sents_pkl2 = os.path.join(preprocessed_root, 'earthquake_train2.pkl')
earthquake_test_sents_pkl2 = os.path.join(preprocessed_root, 'earthquake_test2.pkl')
id_vec_matrix_npy = os.path.join(preprocessed_root, 'id_vec_matrix.npy')

id_vec_matrix = np.load(id_vec_matrix_npy)

def process_pkl(pkl_file):
    f = open(pkl_file, 'r')
    orig_sentences, sentences, discourse_dict = cPickle.load(f)
    f.close()

    A = orig_sentences
    B = id_vec_matrix


    max_sent_len = 0
    for sent in sentences:
        cur_sent_len = len(sent)
        if cur_sent_len > max_sent_len:
            max_sent_len = cur_sent_len

    sentences_matrix = -np.ones((len(sentences), max_sent_len), dtype='int32')
    sentences_matrix_with_start = -np.ones((len(sentences), max_sent_len + 1), dtype='int32')
    for row, sent in enumerate(sentences):
        sentences_matrix[row, 0:len(sent)] = sent
        sentences_matrix_with_start[row, 1:len(sent) + 1] = sent


    C = sentences_matrix
    D = sentences_matrix_with_start

    discourse_list = []
    discourse_list_label = []
    for docname, discourse_list_for_a_doc in discourse_dict.iteritems():
        discourse_list.extend(discourse_list_for_a_doc)

        temp = [0] * len(discourse_list_for_a_doc)
        temp[0] = 1
        discourse_list_label.extend(temp)

    E = discourse_list
    F = discourse_list_label


    file_name, ext = os.path.splitext(pkl_file)
    outfile = file_name + '_final' + ext
    with open(outfile, 'wb') as f:
        cPickle.dump((A, B, C, D, E, F), f, protocol=cPickle.HIGHEST_PROTOCOL)



def write_pkls():
    process_pkl(accident_train_sents_pkl2)
    process_pkl(accident_test_sents_pkl2)
    process_pkl(earthquake_train_sents_pkl2)
    process_pkl(earthquake_test_sents_pkl2)

if __name__ == '__main__':
    write_pkls()