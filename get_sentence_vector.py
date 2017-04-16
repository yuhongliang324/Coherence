__author__ = 'yuhongliang324'
import re, os
from utils import accident_train_root

pattern = re.compile(r'\(\w+? [A-Z]+? "(.+?)"\)')


def get_sentences(root_path):
    files = os.listdir(root_path)
    files.sort()
    doc_grids = {}
    for fn in files:
        if not fn.endswith('.parsed'):
            continue
        sp = fn.split('.')
        doc = '.'.join(sp[:2])
        if doc not in doc_grids:
            doc_grids[doc] = []
        fpath = os.path.join(root_path, fn)
        get_sent(fpath)
        '''
        if grid is None:
            continue
        doc_grids[doc].append(grid)
    return doc_grids'''


def get_sent(parsed_file):
    reader = open(parsed_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    if len(lines) == 0:
        return
    lines = lines[1:-1]

    num_sent = len(lines)
    for i in xrange(num_sent):
        words = re.findall(pattern, lines[i])
        sent = ' '.join(words)
        print sent


get_sentences(accident_train_root)
