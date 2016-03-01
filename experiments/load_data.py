import cPickle as pickle
import numpy as np
import glob
from pylab import *

def read_val_single_moc(net_path, layer):
    '''Reads means, outers, cor for a single net.'''
    with open('%s/val_mean/%s.pkl' % (net_path, layer), 'r') as ff:
        means = pickle.load(ff)
    with open('%s/val_outer/%s.pkl' % (net_path, layer), 'r') as ff:
        outer = pickle.load(ff)
    with open('%s/val_cor/%s.pkl' % (net_path, layer), 'r') as ff:
        cor = pickle.load(ff)
    return means,outer,cor

def read_val_double_cors(net_path, layer, combo):
    '''Reads cor between two nets.'''
    with open('%s/val_cor_%s/%s.pkl' % (net_path, combo, layer), 'r') as ff:
        cor = pickle.load(ff)
    return cor

def glob_and_read(glob_patterns):
    read_paths = []
    for pat in glob_patterns:
        read_paths.extend(glob.glob(pat))
    ims = [imread(pp) for pp in read_paths]
    return ims
    
def vis_for_layer(net_path, layer):
    '''Loads all available vis for whole layer.'''
    
    patterns = []
    patterns.append('%s/unit_vis/%s/%s_montage_maxim.jpg' % (net_path, layer, layer))
    patterns.append('%s/unit_vis/%s/%s_montage_deconv.jpg' % (net_path, layer, layer))
    return glob_and_read(patterns)

def vis_for_unit(net_path, layer, unit_idx):
    '''Loads all available vis for unit.'''
    
    patterns = []
    patterns.append('%s/unit_vis/%s/unit_%04d/montage_maxim.jpg' % (net_path, layer, unit_idx))
    patterns.append('%s/unit_vis/%s/unit_%04d/montage_deconv.jpg' % (net_path, layer, unit_idx))
    return glob_and_read(patterns)

def stacked_vis_for_unit(net_path, layer, unit_idx, vert=True):
    ims = vis_for_unit(net_path, layer, unit_idx)
    return concatenate(ims, axis=0 if vert else 1)


