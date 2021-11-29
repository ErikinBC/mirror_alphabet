# Import modules
import os
import nltk
import string
import spacy
nlp_sm = spacy.load("en_core_web_sm")

import numpy as np
import pandas as pd
import plotnine as pn

from time import time
from scipy import stats
from plydata.cat_tools import *
from funs_support import makeifnot, jaccard, linreg, capture
from funs_cipher import rand_mapping, alpha_trans, get_cipher, n_encipher, annot_vocab_cipher

letters = [l for l in string.ascii_lowercase]

n_letters_seq = np.arange(2,26+1,2).astype(int)
holder = []
for n_letters in n_letters_seq:
    holder.append(n_encipher(n_letters))
df_ncomb = pd.concat(holder).reset_index(drop=True)
print(df_ncomb)


#################################
# --- (1) DOWNLOAD THE DATA --- #

dir_code = os.getcwd()
dir_data = os.path.join(dir_code, '..', 'data')
makeifnot(dir_data)
dir_output = os.path.join(dir_code, '..', 'output')
makeifnot(dir_output)
dir_figures = os.path.join(dir_code, '..', 'figures')
makeifnot(dir_figures)

# (1) Download the ngrams data for frequency
path_ngram = os.path.join(dir_data,'words_ngram.txt')
if not os.path.exists(path_ngram):
    os.system('wget -q -O ../data/words_ngram.txt https://norvig.com/ngrams/count_1w.txt')
else:
    print('words_ngram.txt already exists')

# (2) Download the curated dataset
path_words = os.path.join(dir_data,'words_corncob.txt')
if not os.path.exists(path_words):
    print('Downloading')
    os.system('wget -q -O ../data/words_corncob.txt http://www.mieliestronk.com/corncob_lowercase.txt')
else:
    print('corncob_lowercase.txt already exists')

################################
# --- (2) LOAD THE DATASET --- #

# (1) Load the Ngrams
df_ngram = pd.read_csv(path_ngram,sep='\t',header=None).rename(columns={0:'word',1:'n'})
df_ngram = df_ngram[~df_ngram.word.isnull()].reset_index(None, True)

# (2) Load the short word set
df_words = pd.read_csv(path_words,sep='\n',header=None).rename(columns={0:'word'})
df_words = df_words[~df_words.word.isnull()].reset_index(None, True)

# Overlap?
n_overlap = df_words.word.isin(df_ngram.word).sum()
print('A total of %i short words overlap (out of %i)' % (n_overlap, df_words.shape[0]))

# Merge datasets in the intersection
df_merge = df_ngram.merge(df_words,'inner','word')
df_merge = df_merge.assign(n_sqrt=lambda x: np.sqrt(x.n), n_log=lambda x: np.log(x.n))
# Get the parts of speeach
path_spacy = os.path.join(dir_output, 'pos_spacy.csv')
if os.path.exists(path_spacy):
    pos_spacy = pd.read_csv(path_spacy)
else:
    stime, ncheck = time(), 1000
    holder = []
    for i, txt in enumerate(df_merge.word.to_list()):
        tmp = [pd.DataFrame({'word':txt,'pos':tok.pos_,'tag':tok.tag_},index=[i]) for tok in nlp_sm(txt)][0]
        holder.append(tmp)
        if (i + 1) % ncheck == 0:
            nleft, nrun, dtime = len(df_merge) - (i+1), i + 1, time() - stime
            rate = nrun / dtime
            eta = nleft / rate
            print('%0.1f calculations per second. ETA: %i seconds for %i remaining' %
                  (rate, eta, nleft))
    pos_spacy = pd.concat(holder).reset_index(drop=True)
    # Get the definitions for the different tags
    dat_tag = pd.DataFrame([capture(nltk.help.upenn_tagset,p).split('\n')[0].split(': ') for p in list(pos_spacy.tag.unique())])
    dat_tag.rename(columns={0:'tag',1:'desc'}, inplace=True)
    dat_tag = dat_tag.assign(tag = lambda x: np.where(x.tag.str.len()>5,'XX',x.tag))
    pos_spacy = pos_spacy.merge(dat_tag,'left','tag')
    pos_spacy.to_csv(path_spacy,index=False)

##################################
# --- (3) SUMMARY STATISTICS --- #

# Examine the score frequency by percentiles
p_seq = np.arange(0.01,1,0.01)
dat_n_q = df_merge.melt('word',None,'tt').groupby('tt').value.quantile(p_seq).reset_index()
dat_n_q.rename(columns={'level_1':'qq'}, inplace=True)
dat_n_q.tt = pd.Categorical(dat_n_q.tt,['n','n_sqrt','n_log'])
di_tt = {'n':'n', 'n_sqrt':'sqrt','n_log':'log'}

# DISTIRUBTION OF WORD FREQUENCIES
gg_q = (pn.ggplot(dat_n_q, pn.aes(x='qq',y='value')) + pn.geom_path() +
       pn.theme_bw() + pn.facet_wrap('~tt',scales='free_y') +
       pn.labs(y='Weight', x='Quantile') +
       pn.theme(subplots_adjust={'wspace': 0.25}))
gg_q.save(os.path.join(dir_figures,'gg_q.png'),width=9, height=2.5)

print('The ten most and least common words in the corpus')
print(pd.concat([df_merge.head(10)[['word','n']].reset_index(None,True),
           df_merge.tail(10)[['word','n']].reset_index(None,True)],1))

letter_freq = df_merge['word'].apply(lambda x: list(x),1)
letter_freq = letter_freq.reset_index().explode('word')
letter_freq.rename(columns={'word':'letter','index':'idx'}, inplace=True)
tmp = df_merge.rename_axis('idx')['n'].reset_index()
letter_freq_n = letter_freq.merge(tmp).groupby(['letter']).apply(lambda x: pd.Series({'weight':x['n'].sum(), 'raw':len(x)})).reset_index()
letter_freq_n = letter_freq_n.sort_values('weight',ascending=False).reset_index(drop=True)
print(', '.join(letter_freq_n['letter'].head(12)))


###########################
# --- (4) MODEL CLASS --- #

# class 
from scipy.special import comb

# self=enciphered_dict(df_merge, 'word')


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

    def set_letters(self, letters=None, n_letters=None, idx_letters=None):
        # letters='aZbd';n_letters=None;idx_letters=None
        # letters=None;n_letters=4;idx_letters=10000
        if letters is not None:
            assert isinstance(letters, str), 'Letters needs to be a string'
            self.letters = pd.Series([letter.lower() for letter in letters])
            self.letters = self.letters.drop_duplicates()
            self.n_letters = self.letters.shape[0]
            self.k = int(self.n_letters/2)
            self.letters = np.array(self.letters).reshape([self.k, 2])
            assert self.n_letters % 2 == 0, 'n_letters must be an even number'
            assert self.n_letters <= 26, 'n_letters must be less than or equal to 26'            
        else:
            has_n = n_letters is not None
            has_idx = idx_letters is not None
            assert has_n and has_idx, 'If letters is None, n_letters and idx_letters must be provided'
            self.n_letters = n_letters
            self.k = int(self.n_letters/2)
            self.letters = self.get_lipogram(idx_letters)

    # For a 
    def get_cipher_mat(self, idx):
        n_comb = int(comb(26, self.n_letters))
        assert idx <= n_comb, 'idx must be less than maximum number of combinations: %i' % n_comb
        letters_idx = self.get_comb_idx(idx, 26, self.n_letters)
        letters_lipo = [string.ascii_lowercase[lidx-1] for lidx in letters_idx]
        # Return a k/2 by 2
        res = np.array(letters_lipo).reshape([self.k, 2])
        return res

    def gen_lipo_idx(self, idx):
        n_comb = n_encipher(self.n_letters)['n_lipogram'].values[0]
        assert idx <= n_comb, ''
        j = 0
        lst = self.letters.flatten()
        for i in list(range(self.n_letters-1,0,-2)):
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

    @staticmethod
    def get_comb_idx(idx, n, k):
        # Function which return a list of indices from n choose k permutation
        c, r, j = [], idx, 0
        for s in range(1,k+1):
            cs = j+1
            while r-comb(n-cs,k-s)>0:
                r -= comb(n-cs,k-s)
                cs += 1
            c.append(cs)
            j = cs
        return c



################################
# --- (4) TEST 1:1 MAPPING --- #

# Example of random mapping
num_letters = np.arange(1,27)
di_num2let = dict(zip(num_letters, letters))
nletters = 26
Xmap = rand_mapping(1,letters, nletters)
di_map = dict(zip(Xmap[:,0], Xmap[:,1]))
rwords = df_merge.word.sample(3,random_state=1)
print(pd.DataFrame({'words':rwords, 'mapped':alpha_trans(rwords, Xmap).values}))
print('Mapping: %s' % (','.join(pd.DataFrame(Xmap).apply(lambda x: x.str.cat(sep=':'),1).to_list())))

# Show that our letter pairings are unique
for j in range(2, 13, 2):
    alphabet = letters[0:j]
    jalphabet = ''.join(alphabet)
    mx_perm = npair_max(alphabet)
    print('The first %i letters has %i mappings' % (j, mx_perm))
    malphabet = pd.Series(np.array([alpha_trans([jalphabet],get_cipher(i, alphabet)) for i in range(mx_perm)]).flat)
    assert not malphabet.duplicated().any()

##############################################
# --- (5) RUN ALL CIPHERS FOR 12 LETTERS --- #

alphabet12 = letter_freq_n.letter[0:12].to_list()
jalphabet12 = ''.join(alphabet12)
df_merge12 = df_merge[~df_merge.word.str.contains('[^'+jalphabet12+']')]
words12 = df_merge12.word
print('There are %i unique words using the top 12 letters' % (len(words12)))
words12.reset_index().drop(columns='index').to_csv(os.path.join(dir_output,'words_'+jalphabet12+'.csv'),index=False)

mx_perm12 = npair_max(alphabet12)
di_eval = {0:'idx', 1:'n', 2:'w', 3:'sw', 4:'ln'}

path12 = os.path.join(dir_output,'dat_12.csv')
if os.path.exists(path12):
    dat_12 = pd.read_csv(path12)
else:
    holder = [] # np.zeros(mx_perm12,dtype=int)
    stime, ncheck = time(), 1000
    for i in range(mx_perm12):
        xmap_i = get_cipher(i, alphabet12)
        words_i = words12[alpha_trans(pd.Series(words12), xmap_i).isin(words12)]
        df_i = df_merge12.query('word.isin(@words_i)')
        df_i = pd.DataFrame(np.append([i, len(df_i)],df_i.mean().values)).T.rename(columns=di_eval)
        holder.append(df_i)
        if (i + 1) % ncheck == 0:
            nleft, nrun, dtime = mx_perm12 - (i+1), i + 1, time() - stime
            rate = nrun / dtime
            eta = nleft / rate
            print('%0.1f calculations per second. ETA: %i seconds for %i remaining' %
                  (rate, eta, nleft))
    dat_12 = pd.concat(holder)
    dat_12[['idx','n']] = dat_12[['idx','n']].astype(int)
    dat_12 = dat_12.sort_values('n',ascending=False).reset_index(None,True)
    dat_12.to_csv(path12,index=False)

# "best" word list from each
metric_idx = dat_12.melt('idx',None,'metric').sort_values(['metric','value'],ascending=False)
metric_idx = metric_idx.groupby('metric').head(1).reset_index(None,True)

for i, r in metric_idx.iterrows():
    idx, metric = r['idx'], r['metric']
    vocab_i = annot_vocab_cipher(corpus=words12, cipher=get_cipher(idx, alphabet12), PoS=pos_spacy)
    print('Metric: %s has %i words' % (metric, len(vocab_i)))

# The weighted words aren't helpful
dat_12.drop(columns=['w','sw','ln'], inplace=True)



################################
# --- (6) EVALUATE CIPHERS --- #

di_l2n = dict(zip(letters,range(len(letters))))

# Find statistical enrichments for different letters combinations
holder = []
for i in range(mx_perm12):
    xmap_i = get_cipher(i, alphabet12)
    holder.append(list(pd.DataFrame(xmap_i).apply(lambda x: ':'.join(x),1)))
dat_enrich = pd.DataFrame(holder).assign(n=dat_12.n.values).melt('n',None,'tmp')
dat_enrich = dat_enrich = pd.concat([dat_enrich.drop(columns=['value','tmp']),
           dat_enrich.value.str.split('\\:',1,True).rename(columns={0:'l1',1:'l2'})],1)
dat_enrich = dat_enrich.assign(n1=lambda x: x.l1.map(di_l2n), n2=lambda x: x.l2.map(di_l2n))
dat_enrich = dat_enrich.assign(l1a=lambda x: np.where(x.n1 < x.n2, x.l1, x.l2),
                  l2a=lambda x: np.where(x.n1 < x.n2, x.l2, x.l1)).drop(columns=['n1','n2','l1','l2'])
assert dat_enrich[['l1a','l2a']].apply(lambda x: x.map(di_l2n),0).assign(check=lambda x: x.l1a < x.l2a).check.all()
dat_enrich = dat_enrich.assign(lpair = lambda x: x.l1a + ':' + x.l2a).drop(columns=['l1a','l2a'])

Xbin = pd.get_dummies(dat_enrich.lpair,drop_first=False)
mdl_ols = linreg(inference=True)
mdl_ols.fit(X=Xbin.values,y=dat_enrich.n.values)
dat_ols = pd.DataFrame({'bhat':mdl_ols.bhat,'se':mdl_ols.se}).assign(z=lambda x: x.bhat/x.se)
dat_ols.insert(0,'cn',['Intercept'] + list(Xbin.columns))
dat_ols = dat_ols.assign(is_sig=lambda x: 2*stats.norm.cdf(-np.abs(x.z)) < (0.05/len(x)))
dat_ols = dat_ols.query('cn!="Intercept"').assign(cn=lambda x: cat_reorder(x.cn, x.bhat))

gg_ols = (pn.ggplot(dat_ols, pn.aes(y='cn',x='bhat',color='is_sig')) +
          pn.theme_bw() + pn.geom_point() +
          pn.geom_vline(xintercept=0,linetype='--') +
          pn.labs(x='Coefficient',y='Cipher pairing') +
          pn.scale_color_discrete(name='Statistically significant') +
          pn.theme(legend_position=(0.65,0.25)))
gg_ols.save(os.path.join(dir_figures,'gg_ols.png'), width=5, height=10)


# How rank-correlated are the different measures?
cn_msr = dat_12.columns.drop('idx')
holder = []
for cn1 in cn_msr:
    for cn2 in cn_msr:
        holder.append(pd.DataFrame({'cn1':cn1, 'cn2':cn2, 'rho':stats.spearmanr(dat_12[cn1], dat_12[cn2])[0]},index=[0]))
dat_rho = pd.concat(holder).assign(cn1=lambda x: pd.Categorical(x.cn1, cn_msr),
                                   cn2=lambda x: pd.Categorical(x.cn2, cn_msr))
gg_rho = (pn.ggplot(dat_rho, pn.aes(x='cn1',y='cn2',fill='rho')) +
        pn.theme_bw() + pn.geom_tile(color='black') +
        pn.theme(axis_title=pn.element_blank(),axis_text=pn.element_text(size=12),
              axis_text_x=pn.element_text(angle=90)) +
         pn.scale_fill_gradient2(low='blue',high='red',mid='grey',midpoint=0) +
          pn.guides(fill=False) +
         pn.geom_text(pn.aes(label='rho.round(2)'),color='white') +
         pn.ggtitle("Spearman's rho for ranking metrics"))
gg_rho.save(os.path.join(dir_figures,'gg_rho.png'),width=3, height=3)

# What is the jaccard index for the top 20 ciphers?
k = 20
top_idx = dat_12.head(k).idx.values
holder = []
for i in range(0, k):
    idx_i = top_idx[i]
    xmap_i = get_cipher(idx_i, alphabet12)
    words_i = words12[alpha_trans(pd.Series(words12), xmap_i).isin(words12)]
    for j in range(k):
        idx_j = top_idx[j]
        xmap_j = get_cipher(idx_j, alphabet12)
        words_j = words12[alpha_trans(pd.Series(words12), xmap_j).isin(words12)]
        jac_ij = jaccard(words_i, words_j)
        holder.append(pd.DataFrame({'idx1':idx_i, 'idx2':idx_j, 'jac':jac_ij},index=[0]))
res_jac = pd.concat(holder).reset_index(None,True)
res_jac = res_jac.assign(idx1=lambda x: x.idx1.astype(str),
                         idx2=lambda x: x.idx2.astype(str)).query('idx1 != idx2')
gg_jac = (pn.ggplot(res_jac, pn.aes(x='idx1', y='idx2', fill='jac')) + pn.theme_bw() +
          pn.geom_tile(color='black') +
          pn.labs(y='Cipher #',x='Cipher #') +
          pn.scale_fill_gradient2(name='Jaccard',low='blue',mid='grey',high='red',
                               midpoint=0.2,breaks=np.arange(0,0.51,0.1),limits=(0,0.4)) +
          pn.theme(axis_text_x=pn.element_text(angle=90)) +
          pn.ggtitle('Jaccard index between top 20 ciphers'))
gg_jac.save(os.path.join(dir_figures,'gg_jac.png'),width=4, height=4)


