# General helper functions

import os
import io
import contextlib
import numpy as np

# Capture print outupt
def capture(fun,arg):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        fun(arg)
    output = f.getvalue()
    return output


# Make a folder if it does not exist
def makeifnot(path):
    if not os.path.exists(path):
        print('Path does not exist: %s' % path)
        os.mkdir(path)


# Get the jaccard index between two lists
def jaccard(lst1, lst2):
    return len(np.intersect1d(lst1, lst2)) / len(np.union1d(lst1, lst2))


# Perform a classical OLS regression
class linreg():
    def __init__(self, inference=True):
        self.inference = inference

    def fit(self, X, y):
        n, p = X.shape
        iX = np.c_[np.repeat(1, n), X]
        self.inv_iXX = np.linalg.pinv(iX.T.dot(iX))
        self.Xy = iX.T.dot(y)
        self.bhat = self.inv_iXX.dot(self.Xy)
        residuals = y - iX.dot(self.bhat)
        s2 = np.sum(residuals**2)/(n-(p+1))
        self.se = np.sqrt( s2 * np.diag(self.inv_iXX) )

