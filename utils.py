__author__ = 'yuhongliang324'

import os, re
import numpy
from nltk.stem import WordNetLemmatizer


dn = os.path.dirname(os.path.abspath(__file__))

accident_data_root = os.path.join(dn, 'permutation/accident')
accident_train_root = os.path.join(accident_data_root, 'train')
accident_test_root = os.path.join(accident_data_root, 'test')

earthquake_data_root = os.path.join(dn, 'permutation/earthquake')
earthquake_train_root = os.path.join(earthquake_data_root, 'train')
earthquake_test_root = os.path.join(earthquake_data_root, 'test')

SUBJ, OBJ, OTHER = 3, 2, 1


def load_grid(grid_file):
    reader = open(grid_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    if len(lines) == 0:
        return numpy.zeros((2, 2))

    num_entity = len(lines)
    num_sent = len(lines[0].split()) - 1
    grid = numpy.zeros((num_sent, num_entity))

    for j in xrange(num_entity):
        sps = lines[j].split()
        sps = sps[1:]
        for i in xrange(num_sent):
            if sps[i] == '-':
                continue
            if sps[i] == 'S':
                grid[i, j] = SUBJ
            elif sps[i] == 'O':
                grid[i, j] = OBJ
            else:
                grid[i, j] = OTHER
    return grid


def load(root_path):
    files = os.listdir(root_path)
    files.sort()
    doc_grids = {}
    for fn in files:
        if not fn.endswith('.grid'):
            continue
        sp = fn.split('.')
        doc = '.'.join(sp[:2])
        if doc not in doc_grids:
            doc_grids[doc] = []
        fpath = os.path.join(root_path, fn)
        grid = load_grid(fpath)
        if grid is None:
            continue
        doc_grids[doc].append(grid)
    return doc_grids


def load_verb(root_path):
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
        grid = load_grid_verb(fpath)
        if grid is None:
            continue
        doc_grids[doc].append(grid)
    return doc_grids

pattern = re.compile(r'\(\w+? V[A-Z]*? "(.+?)"\)')
wnl = WordNetLemmatizer()


def load_grid_verb(grid_file):
    reader = open(grid_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    if len(lines) == 0:
        return numpy.zeros((2, 2))
    lines = lines[1:-1]

    num_sent = len(lines)
    verb_ID = {}
    curID = 0
    vIDs = []
    for i in xrange(num_sent):
        vID = set()
        verbs = re.findall(pattern, lines[i])
        verbs = map(lambda x: wnl.lemmatize(x.lower(), pos='v'), verbs)
        for verb in verbs:
            if verb not in verb_ID:
                verb_ID[verb] = curID
                curID += 1
            vID.add(verb_ID[verb])
        vIDs.append(vID)

    num_entity = curID
    grid = numpy.zeros((num_sent, num_entity))

    for i in xrange(num_sent):
        vID = vIDs[i]
        for v in vID:
            grid[i][v] = 1.
    return grid
