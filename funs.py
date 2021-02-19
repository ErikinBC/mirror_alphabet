"""
FUNCTION SUPPORT
"""

import os
import numpy as np
import pandas as pd

def rand_mapping(seed, arr, n):
    np.random.seed(seed)
    return np.random.choice(arr, n, False).reshape([int(n/2),2])
    #timeit('gen_mapping(2,num_letters, nletters)',globals=globals(),number=100000)

# Function to map text based on a two-pair matching
# txt: Any string (or vector
# xmap: a kx2 mapping numpy array
def alpha_trans(txt, xmap):
    assert isinstance(xmap, np.ndarray)
    if not isinstance(txt, pd.Series):
        txt = pd.Series(txt)
    s1 = ''.join(xmap[:,0])
    s2 = ''.join(xmap[:,1])
    trans = str.maketrans(s1+s2, s2+s1)
    return txt.str.translate(trans)

def makeifnot(path):
    if not os.path.exists(path):
        print('Path does not exist: %s' % path)
        os.mkdir(path)

# Function to get the maximum number of mappings for an even-numbered alphabet
def npair_max(lst):
    return np.prod(np.arange(1,len(lst),2))

# Function to get a determistic map of an even-numbered alphabet pairing
# idx: 
def get_cipher(idx,lst):
    assert len(lst) % 2 == 0
    assert idx + 1 <= npair_max(lst)
    lst = lst.copy()
    npair = int(len(lst) / 2)
    holder = np.repeat('1',len(lst)).reshape([npair, 2])
    j = 0
    for i in list(range(len(lst)-1,0,-2)):
        l1 = lst[0]
        q, r = divmod(idx, i)
        r += 1
        l2 = lst[r]
        #print('q: %i, r: %i, l1: %s, l2: %s' % (q, r, l1, l2))
        lst.remove(l1)
        lst.remove(l2)
        holder[j] = [l1, l2]
        j += 1
        idx = q
    return holder
