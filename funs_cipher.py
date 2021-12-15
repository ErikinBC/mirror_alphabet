# Functions to support enciphered alphabet

import string
import numpy as np
import pandas as pd
from scipy.special import comb


"""
df_english:         A DataFrame with a column of words (and other annotations)
cn_word:            Column name in df_english with the English words
letters:            A string of letters (e.g. "abqz")
n_letters:          If letters is None, how many letters to pick from
idx_letters:        If letters is None, which combination index to pick from
"""
class enciphered_dict():
    # df_english=df_merge.copy();cn_word='word'
    def __init__(self, df_english, cn_word):
        assert isinstance(df_english, pd.DataFrame), 'df_english needs to be a DataFrame'
        self.df_english = df_english.rename(columns={cn_word:'word'}).drop_duplicates()
        assert not self.df_english['word'].duplicated().any(), 'Duplicate words found'
        self.df_english['word'] = self.df_english['word'].str.lower()
        self.latin = string.ascii_lowercase
        self.n = len(self.latin)

    """
    After class has been initialized, letters must be chosen. This can be done by either manually specifying the letters, or picking from (26 n_letters)

    letters:        String (e.g. 'aBcd')
    n_letters:      Number of letters to use (must be ≤ 26)
    idx_letters:    When letters is not specified, which of the combination indices to use from (n C k) choices
    """
    def set_letters(self, letters=None, n_letters=None, idx_letters=None):
        # letters='abCd';n_letters=None;idx_letters=None
        # letters=None;n_letters=4;idx_letters=1
        if letters is not None:
            assert isinstance(letters, str), 'Letters needs to be a string'
            self.letters = pd.Series([letter.lower() for letter in letters])
            self.letters = self.letters.drop_duplicates()
            self.letters = self.letters.sort_values().reset_index(drop=True)
            self.n_letters = self.letters.shape[0]
            self.idx_max = {k:v[0] for k,v, in self.n_encipher().to_dict().items()}
        else:
            has_n = n_letters is not None
            has_idx = idx_letters is not None
            assert has_n and has_idx, 'If letters is None, n_letters and idx_letters must be provided'
            self.idx_max = {k:v[0] for k,v, in self.n_encipher().to_dict().items()}
            assert idx_letters <= self.idx_max['n_lipogram'], 'idx_letters must be ≤ %i' % self.idx_max['n_lipogram']
            assert idx_letters > 0, 'idx_letters must be > 0'
            self.n_letters = n_letters
            tmp_idx = self.get_comb_idx(idx_letters, self.n, self.n_letters)
            self.letters = pd.Series([self.latin[idx-1] for idx in tmp_idx])
            self.letters = self.letters.sort_values().reset_index(drop=True)
        assert self.n_letters % 2 == 0, 'n_letters must be an even number'
        assert self.n_letters <= self.n, 'n_letters must be ≤ %i' % self.n
        self.k = int(self.n_letters/2)
        
    
    """
    After letters have been set, either specify mapping or pick from an index

    pairing:        String specifying pairing order (e.g. 'a:e, i:o')
    idx_pairing:    If the pairing is not provided, pick one of the 1 to n_encipher possible permutations
    """
    def set_encipher(self, pairing=None, idx_pairing=None):
        # pairing='a:b, c:d';idx_pairing=None
        # pairing=None;idx_pairing=3
        if pairing is not None:
            assert isinstance(pairing, str), 'pairing needs to be a string'
            lst_pairing = pairing.replace(' ','').split(',')
            self.mat_pairing = np.array([pair.split(':') for pair in lst_pairing])
            assert self.k == self.mat_pairing.shape[0], 'number of rows does not equal k: %i' % self.k
            assert self.mat_pairing.shape[1] == 2, 'mat_pairing does not have 2 columns'
            tmp_letters = self.mat_pairing.flatten()
            n_tmp = len(tmp_letters)
            assert n_tmp == self.n_letters, 'The pairing list does not match number of letters: %i to %i' % (n_tmp, self.n_letters)
            lst_miss = np.setdiff1d(self.letters, tmp_letters)
            assert len(lst_miss) == 0, 'pairing does not have these letters: %s' % lst_miss
        else:
            assert idx_pairing > 0, 'idx_pairing must be > 0'
            assert idx_pairing <= self.idx_max['n_encipher'], 'idx_pairing must be ≤ %i' % self.idx_max['n_encipher']
            # Apply determinstic formula
            self.mat_pairing = self.get_encipher_idx(idx_pairing)
        # Pre-calculated values for alpha_trans() method
        s1 = ''.join(self.mat_pairing[:,0])
        s2 = ''.join(self.mat_pairing[:,1])
        self.trans = str.maketrans(s1+s2, s2+s1)

    """
    Find enciphered corpus
    """
    def get_corpus(self):
        words = self.df_english['word']
        # Remove words that have a letter outside of the lipogram
        regex_lipo = '[^%s]' % ''.join(self.letters)
        words = words[~words.str.contains(regex_lipo)].reset_index(drop=True)
        words_trans = self.alpha_trans(words)
        idx_match = words.isin(words_trans)
        tmp1 = words[idx_match]
        tmp2 = words_trans[idx_match]
        self.df_encipher = pd.DataFrame({'word':tmp1,'mirror':tmp2})
        self.df_encipher.reset_index(drop=True,inplace=True)
        # Add on any other columns from the original dataframe
        self.df_encipher = self.df_encipher.merge(self.df_english)

    """
    Deterministically returns encipher
    """
    def get_encipher_idx(self, idx):
        j = 0
        lst = self.letters.to_list()
        holder = np.repeat('1',self.n_letters).reshape([self.k, 2])
        for i in list(range(self.n_letters-1,0,-2)):
            l1 = lst[0]
            q, r = divmod(idx, i)
            r += 1
            l2 = lst[r]
            lst.remove(l1)
            lst.remove(l2)
            holder[j] = [l1, l2]
            j += 1
            idx = q
        return holder

    """
    Deterministically return (n C k) indices
    """
    @staticmethod
    def get_comb_idx(idx, n, k):
        c, r, j = [], idx, 0
        for s in range(1,k+1):
            cs = j+1
            while r-comb(n-cs,k-s)>0:
                r -= comb(n-cs,k-s)
                cs += 1
            c.append(cs)
            j = cs
        return c

    """
    Uses mat_pairing to translate the strings

    txt:        Any string or Series
    """
    def alpha_trans(self, txt):
        # txt='abod';txt=pd.Series(['abcd','dcba'])
        if not isinstance(txt, pd.Series):
            txt = pd.Series(txt)
        z = txt.str.translate(self.trans)
        return z

    """
    Function to calculate total number lipogrammatic and enciphering combinations
    """
    def n_encipher(self, n_letters=None):
        if n_letters is None:
            n_letters = self.n_letters
        assert n_letters % 2 == 0, 'n_letters is not even'
        n1 = int(np.prod(np.arange(1,n_letters,2)))
        n2 = int(comb(26, n_letters))
        n_tot = n1 * n2
        res = pd.DataFrame({'n_letter':n_letters,'n_encipher':n1, 'n_lipogram':n2, 'n_total':n_tot},index=[0])
        return res

