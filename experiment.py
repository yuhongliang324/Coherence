__author__ = 'yuhongliang324'

from utils import accident_test_root, earthquake_test_root, load, load_verb
from graph import get_coherence
import numpy


def test_graph(test_root, ent=True, verb=False):
    doc_coherences = None
    if ent:
        doc_grids = load(test_root)
        doc_coherences = calculate_coherences(doc_grids)
    if verb:
        doc_grids = load_verb(test_root)
        doc_coherences_verb = calculate_coherences(doc_grids)
        if doc_coherences is None:
            doc_coherences = doc_coherences_verb
        else:
            docs = doc_coherences.keys()
            for doc in docs:
                doc_coherences[doc] += doc_coherences_verb[doc]
    right, total = 0., 0.
    coherences = doc_coherences.values()
    for coh in coherences:
        num_coh = len(coh)
        for i in xrange(1, num_coh):
            if coh[0] > coh[i]:
                right += 1
            elif coh[0] == coh[i]:
                right += 0.5
            total += 1
    print right, total, right / float(total)


def calculate_coherences(doc_grids):
    doc_coherences = {}
    for doc, grids in doc_grids.items():
        cohs = []
        for grid in grids:
            coh = get_coherence(grid)
            cohs.append(coh)
        doc_coherences[doc] = numpy.asarray(cohs)
    return doc_coherences


test_graph(accident_test_root, ent=False, verb=True)
