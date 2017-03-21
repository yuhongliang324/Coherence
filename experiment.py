__author__ = 'yuhongliang324'

from utils import accident_test_root, earthquake_test_root, load
from graph import get_coherence


def test1():
    doc_grids = load(earthquake_test_root)
    grids2 = doc_grids.values()
    right, total = 0, 0
    for grids in grids2:
        coherences = []
        for grid in grids:
            coh = get_coherence(grid)
            coherences.append(coh)
        ng = len(coherences)
        for i in xrange(1, ng):
            if coherences[0] > coherences[i]:
                right += 1
            total += 1
    print right, total, right / float(total)


test1()
