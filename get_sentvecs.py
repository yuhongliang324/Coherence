__author__ = 'yuhongliang324'
import cPickle, os
import skipthoughts

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)


preprocessed_root = '/usr0/home/hongliay/code/Coherence/preprocessed'
accident_train_sents_pkl = os.path.join(preprocessed_root, 'accident_train.pkl')
accident_test_sents_pkl = os.path.join(preprocessed_root, 'accident_test.pkl')
earthquake_train_sents_pkl = os.path.join(preprocessed_root, 'earthquake_train.pkl')
earthquake_test_sents_pkl = os.path.join(preprocessed_root, 'earthquake_test.pkl')


accident_train_vecs_pkl = os.path.join(preprocessed_root, 'accident_vecs_train.pkl')
accident_test_vecs_pkl = os.path.join(preprocessed_root, 'accident_vecs_test.pkl')
earthquake_train_vecs_pkl = os.path.join(preprocessed_root, 'earthquake_vecs_train.pkl')
earthquake_test_vecs_pkl = os.path.join(preprocessed_root, 'earthquake_vecs_test.pkl')


def get_vec(sents_pkl, out_pkl):
    reader = open(sents_pkl)
    doc_sents_list = cPickle.load(reader)
    reader.close()

    doc_vecs_list = {}

    for doc, sents_list in doc_sents_list.items():
        vecs_list = []
        for sents in sents_list:
            vecs = encoder.encode(sents)
            vecs_list.append(vecs)
        doc_vecs_list[doc] = vecs_list

    f = open(out_pkl, 'wb')
    cPickle.dump(doc_vecs_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    print 'Dumped to ' + out_pkl


if __name__ == '__main__':
    get_vec(accident_train_sents_pkl, accident_train_vecs_pkl)
    get_vec(accident_test_sents_pkl, accident_test_vecs_pkl)
    get_vec(earthquake_train_sents_pkl, earthquake_train_vecs_pkl)
    get_vec(earthquake_test_sents_pkl, earthquake_test_vecs_pkl)
