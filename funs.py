"""
FUNCTION SUPPORT
"""

import os
import numpy as np
import pandas as pd

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


def jaccard(lst1, lst2):
    return len(np.intersect1d(lst1, lst2)) / len(np.union1d(lst1, lst2))


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


# Function that takes a word list, cipher, and returns word pairings with word types
# corpus: some vector of strings
# cipher: numpy kx2 cipher array
# PoS: parts of speech lookup
def annot_vocab_cipher(corpus, cipher, PoS):
    assert isinstance(cipher, np.ndarray)
    if not isinstance(corpus,pd.Series):
        corpus = pd.Series(corpus.copy())
    tcorpus = alpha_trans(corpus, cipher)
    idx = corpus.isin(tcorpus)
    corpus, tcorpus = corpus[idx], tcorpus[idx]
    df = pd.DataFrame({'word':corpus,'mirror':tcorpus})
    df = df.merge(PoS,'left','word').sort_values('pos').reset_index(None, True)
    df = df.merge(PoS,'left',left_on='mirror',right_on='word')#.drop(columns=['word_y','desc_y'])
    cn_ord = ['word_x','word_y','pos_x','pos_y','tag_x','tag_y','desc_x']
    df = df[cn_ord]
    return df



def makeifnot(path):
    if not os.path.exists(path):
        print('Path does not exist: %s' % path)
        os.mkdir(path)

# Function to get the maximum number of mappings for an even-numbered alphabet
def npair_max(lst):
    return np.prod(np.arange(1,len(lst),2))

# Function to get a determistic map of an even-numbered alphabet pairing
# idx: an integer less than the max number of combinations
# lst: an even numbered list of unique elements which will be mapped
def get_cipher(idx,lst):
    assert len(lst) % 2 == 0
    assert idx + 1 <= npair_max(lst)
    assert not pd.Series(lst).duplicated().any()
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
