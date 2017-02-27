# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import h5py
from scipy.io import loadmat

def load_score_mat(filename, data_name):
    """this function read image search scores in mat file"""
    if os.path.exists(filename):
        mat_file = h5py.File(filename)
        if data_name not in mat_file.keys():
            print "%s element does not exists" % data_name
            print 'mat file has keys:', mat_file.keys()
            sys.exit()
        data = mat_file[data_name].value
        return np.array(data)
    else:
        print "%s file does not exists" % filename
        sys.exit()


def load_mat(filename, data_name):
    if os.path.exists(filename):
        mat_file = loadmat(filename)
        if data_name not in mat_file.keys():
            print "%s element does not exists" % data_name
            print 'mat file has keys:', mat_file.keys()
            sys.exit()
        data = mat_file[data_name]
        return np.array(data)
    else:
        print "%s file does not exists" % filename
        sys.exit()


def load_score_txt(filename):
    """this function read image search scores in txt file"""
    if os.path.exists(filename):
        feature_score = np.fromfile(filename, dtype=np.float32)
        return feature_score
    else:
        print "%s file does not exists" % filename
        sys.exit()


def reshape_score(data, shape):
    """this function to reshape the score"""
    if isinstance(data, np.ndarray):
        data = data.reshape(shape)
        return data
    else:
        print "shape error"
        sys.exit()
