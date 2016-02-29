import numpy as np
from pycache import memoize_no_func_body
from numpy import array as array
import networkx as nx


def permute_matrix(mat, new_order):
    new_order = array(new_order)
    ret = mat[new_order,:][:,new_order]
    assert np.all(abs(np.sort(mat.sum(0)) - np.sort(ret.sum(0))) < 1.0), 'sanity check'
    assert np.all(abs(np.sort(mat.sum(1)) - np.sort(ret.sum(1))) < 1.0), 'sanity check'
    return ret

def permute_rows(mat, new_order):
    new_order = array(new_order)
    ret = mat[new_order,:]
    assert np.all(abs(np.sort(mat.sum(0)) - np.sort(ret.sum(0))) < 1.0), 'sanity check'
    assert np.all(abs(np.sort(mat.sum(1)) - np.sort(ret.sum(1))) < 1.0), 'sanity check'
    return ret

def permute_cols(mat, new_order):
    new_order = array(new_order)
    ret = mat[:,new_order]
    assert np.all(abs(np.sort(mat.sum(0)) - np.sort(ret.sum(0))) < 1.0), 'sanity check'
    assert np.all(abs(np.sort(mat.sum(1)) - np.sort(ret.sum(1))) < 1.0), 'sanity check'
    return ret

def _max_match_order(mat, ignore_diag = False):
    # build bipartite graph
    gg = nx.Graph()
    assert mat.shape[0] == mat.shape[1]
    size = mat.shape[0]
    for ii in xrange(size):
        for jj in xrange(size):
            if ignore_diag and ii == jj:
                continue
            gg.add_edge(ii, jj+size, weight=mat[ii,jj])

    matching = nx.max_weight_matching(gg, maxcardinality=True)
    order = array([matching[ii]-size for ii in xrange(size)])
    return order

@memoize_no_func_body
def max_match_order(mat, ignore_diag = False):
    return _max_match_order(mat, ignore_diag = ignore_diag)

def _max_match_order_m(mat, ignore_diag = False):
    # build bipartite graph
    assert mat.shape[0] == mat.shape[1]
    size = mat.shape[0]
    mat = -mat.copy()
    if ignore_diag:
        # Set diagonal values to be higher than the max so they are not chosen
        mat[arange(size), arange(size)] = mat.max() + 1
    munk = Munkres()
    path = munk.compute(mat)
    
    assert len(path) == size
    order = arange(size) * 0
    for ii, (rr,cc) in enumerate(path):
        assert ii == rr
        order[rr] = cc
        if ignore_diag:
            assert rr != cc
        
    return order

@memoize_no_func_body
def max_match_order_m(mat, ignore_diag = False):
    return _max_match_order_m(mat, ignore_diag = ignore_diag)


def follow_loops(order):
    '''order is that given by max_match_order'''
    seen = set()
    ret = []
    lengths = []
    while len(seen) < len(order):
        # Find first new element
        for ii in range(len(order)):
            if ii not in seen:
                idx_start = ii
                break
        this_length = 1
        ret.append(idx_start)
        seen.add(idx_start)
        ii = order[idx_start]
        while ii != idx_start:
            ret.append(ii)
            seen.add(ii)
            ii = order[ii]
            this_length += 1
        lengths.append(this_length)
    return ret,lengths
