
import cPickle, os
from utils import dn

preprocessed_root = os.path.join(dn, 'preprocessed')
accident_train_sents_pkl3_final = os.path.join(preprocessed_root, 'accident_train3_final.pkl')
accident_test_sents_pkl3_final = os.path.join(preprocessed_root, 'accident_test3_final.pkl')
earthquake_train_sents_pkl3_final = os.path.join(preprocessed_root, 'earthquake_train3_final.pkl')
earthquake_test_sents_pkl3_final = os.path.join(preprocessed_root, 'earthquake_test3_final.pkl')


def check_if_ascend(final_pkl):
    with open(final_pkl, 'rb') as f:
        _, _, _, _, discourse_list, discourse_list_label, _, _, _, _, _, _ = cPickle.load(f)

    for row in xrange(len(discourse_list_label)):
        if discourse_list_label[row] != 1:
            continue

        cur_discourse = discourse_list[row]

        for idx in xrange(len(cur_discourse) - 1):
            if cur_discourse[idx] > cur_discourse[idx + 1]:
                print cur_discourse
                return False

    return True


def check():
    print check_if_ascend(accident_train_sents_pkl3_final)
    print check_if_ascend(accident_test_sents_pkl3_final)
    print check_if_ascend(earthquake_train_sents_pkl3_final)
    print check_if_ascend(earthquake_test_sents_pkl3_final)


def kendall_tau(discourse_gt, discourse_sample, first_sent_always_correct=True):
    if first_sent_always_correct:
        if discourse_gt[0] == discourse_sample[0]:
            discourse_gt = discourse_gt[1:]
            discourse_sample = discourse_sample[1:]
        else:
            return -1.0

    if len(discourse_gt) != len(discourse_sample):
        return -1.0
    if set(discourse_gt) != set(discourse_sample):
        return -1.0

    intersection = 0
    for idx in xrange(len(discourse_gt)):
        sent = discourse_gt[idx]
        target_idx = discourse_sample.index(sent)

        intersection += target_idx - idx

        discourse_sample.remove(sent)
        discourse_sample.insert(idx, sent)

    print 'discourse_gt:', discourse_gt
    print 'discourse_sample:', discourse_sample

    N = len(discourse_gt)

    return 1 - (2.0 * intersection) / (N * (N - 1))


if __name__ == '__main__':
    # check() # Failed. There are discourses that have non-ascending sentence indeces.
    print kendall_tau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    print kendall_tau([1, 2, 3, 4, 5], [1, 4, 5, 2, 3])
    print kendall_tau([1, 2, 3, 4, 5], [1, 4, 3, 2, 5])
