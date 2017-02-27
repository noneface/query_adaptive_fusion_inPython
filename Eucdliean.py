# -*-coding:utf-8 -*-

import sys
import numpy as np


def dist(x, c):
    """eucdliean distance == (a-b)^2 == (a^2 + b^2 - 2*a*b)"""
    ndata, dimx = x.shape
    ncentres, dimc = c.shape

    if dimx != dimc:
        print "Data dimension does not match dimension of centres"
        sys.exit()

    a = np.dot(np.ones((ncentres, 1)), np.sum(x**2, axis=1).reshape(1, -1)).T
    b = np.dot(np.ones((ndata, 1)), np.sum(c**2, axis=1).reshape(1, -1))
    c = np.dot(x,c.T)

    n2 = a + b - 2*c
    return n2
