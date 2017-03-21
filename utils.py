__author__ = 'yuhongliang324'

import os
import numpy


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
        return None

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
