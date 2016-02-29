import numpy as np
import matplotlib.pyplot as plt

"""helper functions for estimating mutual information"""

def estimate_mi(hist2d):
    '''Estimate MI from 2D histogram'''
    num_samples = hist2d.sum()
    pij = hist2d / float(num_samples)
    pi = pij.sum(1)
    pj = pij.sum(0)
    mi = 0.0
    for ii in range(pi.shape[0]):
        for jj in range(pj.shape[0]):
            if pij[ii,jj] > 0:
                mi += pij[ii,jj] * log(pij[ii,jj] / (pi[ii]*pj[jj]))
                #print ii, jj, mi
    approx_bias = float(prod(pij.shape)) / (2 * num_samples * log(2))
    return mi, mi - approx_bias

def percentile_bins(xy, num_bins = 10):
    x_zero_fraction = float(sum(xy[:,0] < 1e-6)) / xy.shape[0]
    x_percentiles = [0] + list(linspace(x_zero_fraction * 100.1,100,num_bins))
    x_edges = np.percentile(xy[:,0], x_percentiles)
    x_edges[0] -= .01
    
    y_zero_fraction = float(sum(xy[:,1] < 1e-6)) / xy.shape[0]
    y_percentiles = [0] + list(linspace(y_zero_fraction * 100.1,100,num_bins))
    y_edges = np.percentile(xy[:,1], y_percentiles)
    y_edges[0] -= .01

    return x_edges, y_edges

def plot_mi(xy, bins = 10):
    hist2d, xedges, yedges = histogram2d(xy[:,0], xy[:,1], bins=bins)
    assert abs(xy.shape[0] - hist2d.sum()) < .01, 'some data fell outside bins'
    cor = corrcoef(xy[:,0], xy[:,1])[0,1]
    mi,mi_lessbias = estimate_mi(hist2d)

    ax = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
    
    X, Y = np.meshgrid(xedges, yedges)
    white_val = np.percentile(hist2d.flatten(), 99)  # saturate earlier so plot isn't all black
    ax.pcolormesh(X, Y, np.minimum(hist2d, white_val))
    axis('tight')
    title('Cor: %.4f, MI: %.4f, MI less bias: %.4f' % (cor, mi, mi_lessbias))

    ax = plt.subplot2grid((2,3), (0,2))
    ax.bar(xedges[:-1], hist2d.sum(1))
    title('x bin counts')
    
    ax = plt.subplot2grid((2,3), (1,2))
    ax.bar(yedges[:-1], hist2d.sum(0))
    title('y bin counts')

