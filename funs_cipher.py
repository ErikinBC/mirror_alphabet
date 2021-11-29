# Functions to support enciphered alphabet

import numpy as np
import pandas as pd
from scipy.special import comb

# Function to get the maximum number of mappings for an even-numbered alphabet
def n_encipher(n_letters):
    assert n_letters % 2 == 0, 'n_letters is not even'
    n1 = int(np.prod(np.arange(1,n_letters,2)))
    n2 = int(comb(26, n_letters))
    n_tot = n1 * n2
    res = pd.DataFrame({'n_letter':n_letters,'n_encipher':n1, 'n_lipogram':n2, 'n_total':n_tot},index=[0])
    return res

# Generate a random mapping of.....
def rand_mapping(seed, arr, n):
    np.random.seed(seed)
    return np.random.choice(arr, n, False).reshape([int(n/2),2])


"""
Function to map text based on a two-pair matching

txt:        Any string (or vector)
xmap:       A kx2 mapping numpy array
"""
def alpha_trans(txt, xmap):
    assert isinstance(xmap, np.ndarray)
    if not isinstance(txt, pd.Series):
        txt = pd.Series(txt)
    s1 = ''.join(xmap[:,0])
    s2 = ''.join(xmap[:,1])
    trans = str.maketrans(s1+s2, s2+s1)
    return txt.str.translate(trans)


"""
Function that takes a word list, cipher, and returns word pairings with word types

corpus:         Vvector of strings
cipher:         Numpy kx2 cipher array
PoS:            Parts of speech lookup
"""
def annot_vocab_cipher(corpus, cipher, PoS):
    assert isinstance(cipher, np.ndarray), 'cipher is not an np.ndarray'
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




"""
Function to get a determistic map of an even-numbered alphabet pairing

idx:        An integer less than the max number of combinations
lst:        An even numbered list of unique elements which will be mapped
"""
def get_cipher(idx,lst):
    assert len(lst) % 2 == 0, 'lst is not an even numbered length'
    assert idx + 1 <= npair_max(lst)
    assert not pd.Series(lst).duplicated().any(), 'Duplicates in lst found'
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
