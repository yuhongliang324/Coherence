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


def get_grid_dict(token_file):
    token_file_list = token_file.split('/')
    filename = os.path.splitext(token_file_list[-1])[0] + '.parsed.grid'
    dataset = '/'.join(token_file_list[-2].split('_'))

    permutation_root = os.path.join(dn, 'permutation')

    grid_file = os.path.join(permutation_root, dataset, filename)

    grid_dict = {}
    with open(grid_file, 'r') as f:
        for line in f:
            content = line.rstrip().split()
            grid_dict[content[0].lower()] = content[1:]

    return grid_dict


def get_role_codes(token_file, pos_file):
    grid_dict = get_grid_dict(token_file)
    # B_set = set(['be', 'am', 'is', 'are', 'was', 'were', 'being', 'been']) # might add more "stop verbs" into this list!
    B_set = set(['be', 'am', 'is', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'having']) # might add more "stop verbs" into this list!

    token_f = open(token_file)
    pos_f = open(pos_file)

    token_lines = token_f.readlines()
    pos_lines = pos_f.readlines()

    token_f.close()
    pos_f.close()

    if len(token_lines) == 0:
        return None

    num_sent = len(token_lines)
    para_role_codes = []

    for i in xrange(num_sent):
        sent_role_codes = []

        tokens = token_lines[i].rstrip().split()
        poss = pos_lines[i].rstrip().split()

        for j in xrange(len(tokens)):
            token = tokens[j]
            pos = poss[j]

            if pos.startswith('v'):
                if token in B_set:
                    sent_role_codes.append('B')
                else:
                    sent_role_codes.append('V')
            elif pos.startswith('n'):
                if token in grid_dict:
                    role = grid_dict[token][i]
                    if role == '-':
                        print 'token: %s, pos: %s, role: -' % (token, pos)
                        sent_role_codes.append('?')
                    else:
                        sent_role_codes.append(role)
                else:
                    # print 'token_file: %s\npos_file: %s' % (token_file, pos_file)
                    print 'token: %s, pos: %s, startswith N but not in the grid_dict!' % (token, pos)
                    sent_role_codes.append('?')
            else:
                sent_role_codes.append('?')

        para_role_codes.append(' '.join(sent_role_codes))
    return para_role_codes


def write_all_role_codes(root_path, dataset, out_root):
    if not os.path.isdir(out_root):
        os.mkdir(out_root)
    files = os.listdir(root_path)
    files.sort()
    for fn in files:
        if not fn.endswith('.parsed'):
            continue
        
        fn = fn.replace('parsed', 'txt')
        token_file = os.path.join(preprocessed_root, dataset, fn)
        pos_file = os.path.join(preprocessed_root, dataset + '_pos', fn)

        role_codes = get_role_codes(token_file, pos_file)
        # sents = get_sents(os.path.join(root_path, fn))
        writer = open(os.path.join(out_root, fn), 'w')
        if role_codes is None:
            writer.close()
            continue
        for role_code in role_codes:
            writer.write(role_code[:-1] + '?\n')
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

def write_role_codes():
    write_all_role_codes(accident_train_root, 'accident_train', accident_train_sents_root + '_role_code')
    write_all_role_codes(accident_test_root, 'accident_test', accident_test_sents_root + '_role_code')
    write_all_role_codes(earthquake_train_root, 'earthquake_train', earthquake_train_sents_root + '_role_code')
    write_all_role_codes(earthquake_test_root, 'earthquake_test', earthquake_test_sents_root + '_role_code')



if __name__ == '__main__':
    # test3()
    write_role_codes()

