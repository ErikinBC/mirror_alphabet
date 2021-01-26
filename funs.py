"""
FUNCTION SUPPORT
"""

import os
import numpy as np

def rand_mapping(seed, arr, n):
  np.random.seed(seed)
  return np.random.choice(arr, n, False).reshape([int(n/2),2])
#timeit('gen_mapping(2,num_letters, nletters)',globals=globals(),number=100000)

def alpha_trans(txt, xmap):
  s1 = ''.join(xmap[:,0])
  s2 = ''.join(xmap[:,1])
  trans = str.maketrans(s1+s2, s2+s1)
  return txt.str.translate(trans)

def makeifnot(path):
    if not os.path.exists(path):
        print('Path does not exist: %s' % path)
        os.mkdir(path)


