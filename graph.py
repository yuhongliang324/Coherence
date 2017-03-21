__author__ = 'yuhongliang324'
import numpy


def get_adjacency_matrix(grid, weight='Acc'):
    weight = weight.lower()
    if weight != 'acc':
        grid[grid > 0] = 1.  # (n_sent, n_entity)
    A = numpy.dot(grid, grid.T)
    if weight == 'u':
        A[A > 0] = 1.
    numpy.fill_diagonal(A, 0.)
    return numpy.triu(A)


def get_coherence_from_adjacency(A):
    n_sent = A.shape[0]
    for i in xrange(n_sent - 1):
        for j in xrange(i + 2, n_sent):
            A[i, j] /= (j - i)
    coherence = numpy.mean(numpy.sum(A, axis=1))
    return coherence


def get_coherence(grid, weight='Acc'):
    A = get_adjacency_matrix(grid, weight=weight)
    coherence = get_coherence_from_adjacency(A)
    return coherence
