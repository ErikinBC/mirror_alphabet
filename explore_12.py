
# Import modules
import os
import nltk
import string
import spacy
nlp_sm = spacy.load('en_core_web_sm')

import numpy as np
import pandas as pd
import plotnine as pn

from time import time
from scipy import stats
from plydata.cat_tools import *
from funs_support import makeifnot, jaccard, linreg, capture
from funs_cipher import encipherer

letters = [l for l in string.ascii_lowercase]

n_letters_seq = np.arange(2,26+1,2).astype(int)
holder = []
for n_letters in n_letters_seq:
    holder.append(encipherer.n_encipher(n_letters))
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
df_ngram = df_ngram[~df_ngram['word'].isnull()].reset_index(drop=True)

# (2) Load the short word set
df_words = pd.read_csv(path_words,sep='\n',header=None).rename(columns={0:'word'})
df_words = df_words[~df_words['word'].isnull()].reset_index(drop=True)

# Overlap?
n_overlap = df_words.word.isin(df_ngram['word']).sum()
print('A total of %i short words overlap (out of %i)' % (n_overlap, df_words.shape[0]))

# Merge datasets in the intersection
df_merge = df_ngram.merge(df_words,'inner','word')
df_merge = df_merge.assign(n_sqrt=lambda x: np.sqrt(x['n']), n_log=lambda x: np.log(x['n']))

# Add on the parts of speech
pos_lst = [z[1] for z in nltk.pos_tag(df_merge['word'].to_list())]
df_merge.insert(1,'pos',pos_lst)
# Get PoS defintions
pos_def = pd.Series([capture(nltk.help.upenn_tagset,p) for p in df_merge['pos'].unique()])
pos_def = pos_def.str.split('\\:\\s|\\n',expand=True,n=3).iloc[:,:2]
pos_def.rename(columns={0:'pos',1:'def'},inplace=True)
df_merge = df_merge.merge(pos_def, 'left', 'pos')


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
top12_letters = letter_freq_n['letter'].head(12)
print(', '.join(top12_letters))

#############################
# --- (4) SANITY CHECKS --- #

# Do the sanity checks
enc = encipherer(df_merge, 'word')
n_lipogram = enc.n_encipher(n_letters=4)['n_lipogram'][0]

# (i) Enumerate through all possible letter pairings
holder = []
for i in range(1, n_lipogram+1):
    if i % 500 == 0:
        print('Iteration %i of %i' % (i, n_lipogram))
    enc.set_letters(n_letters=4, idx_letters=i)
    holder.append(enc.letters)
df_letters = pd.DataFrame(holder)
df_letters.columns = ['l'+str(i+1) for i in range(4)]
assert not df_letters.duplicated().any()  # Check that no duplicate values
df_letters

# (ii) Enumerate through all possible ciphers
enc = encipherer(df_merge, 'word')
enc.set_letters(n_letters=12, idx_letters=1)
n_encipher = enc.n_encipher(enc.n_letters)['n_encipher'][0]

holder = []
for i in range(1, n_encipher+1):
    if i % 500 == 0:
        print('Iteration %i of %i' % (i, n_encipher))
    enc.set_encipher(idx_pairing=i)
    holder.append(enc.mat_pairing.flatten())
df_encipher = pd.DataFrame(holder)
idx_even = df_encipher.columns % 2 == 0
tmp1 = df_encipher.loc[:,idx_even]
tmp2 = df_encipher.loc[:,~idx_even]
tmp2.columns = tmp1.columns
df_encipher = tmp1 + ':' + tmp2
df_encipher.columns = ['sub'+str(i+1) for i in range(6)]
assert not df_encipher.duplicated().any()  # Check that no duplicate values
df_encipher


# (ii) Enumerate through all possible pairing/ciphers
enc = encipherer(df_merge, 'word')
enc.set_letters(n_letters=4, idx_letters=1)
n_encipher = enc.idx_max['n_encipher']
n_lipogram = enc.idx_max['n_lipogram']
n_total = enc.idx_max['n_total']
holder = pd.Series(np.repeat('',n_total))
k = 0
for j in range(1, n_lipogram+1):
    if j % 500 == 0:
        print('Iteration %i of %i' % (j, n_lipogram))
    enc.set_letters(n_letters=4, idx_letters=j)
    for i in range(1, n_encipher+1):
        enc.set_encipher(idx_pairing=i)
        res = pd.Series(enc.str_pairing)
        holder[k] = res
        k += 1
res_all = holder.reset_index(drop=True).apply(lambda x: x[0])
assert not res_all.duplicated().any()

# (iv) High-quality mapping
enc = encipherer(df_merge, 'word')
enc.set_letters(letters='etoaisnrlchd')
enc.set_encipher(idx_pairing=1)
enc.get_corpus()
print(enc.str_pairing)
enc.df_encipher[['word','mirror','pos','def']]

##############################
# --- (5) CIPHER QUALITY --- #

# (i) Score the ciphers
enc = encipherer(df_merge, 'word')
enc.set_letters(letters='etoaisnrlchd')
enc.score_ciphers(cn_weight='n_log',set_best=True)

# (ii) Visualize the distribution
gg_score_w = (pn.ggplot(enc.df_score, pn.aes(x='n_word',y='weight')) + 
    pn.theme_bw() + pn.geom_point() + 
    pn.ggtitle('Relationship between word count and score') + 
    pn.geom_smooth(method='lm',se=False) + 
    pn.labs(y='Weight', x='# of words'))
gg_score_w.save(os.path.join(dir_figures,'gg_score_w.png'),width=5,height=3.5)

long_score = enc.df_score.melt('idx',None,'metric')
di_metric = {'n_word':'# of words', 'weight':'Weight'}
gg_dist_score = (pn.ggplot(long_score, pn.aes(x='value',fill='metric')) + 
    pn.theme_bw() + pn.guides(fill=False) + 
    pn.geom_histogram(bins=25,color='black') + 
    pn.ggtitle('Distribution of score and word count') + 
    pn.facet_wrap('~metric',labeller=pn.labeller(metric=di_metric),scales='free_x') + 
    pn.theme(subplots_adjust={'wspace': 0.20}, axis_title_x=pn.element_blank()) + 
    pn.labs(y='Frequency'))
gg_dist_score.save(os.path.join(dir_figures,'gg_dist_score.png'),width=9,height=4)


# Add on the cipher pairs to be saved for later
enc2 = encipherer(df_merge, 'word')
enc2.set_letters(letters='etoaisnrlchd')
holder = []
for i in range(1, n_encipher+1):
    if (i + 1) % 500 == 0:
        print(i+1)
    enc2.set_encipher(idx_pairing=i)
    holder.append(enc2.str_pairing)
dat_mapping = pd.Series(holder).str.split(',',5,True)
dat_mapping = pd.get_dummies(dat_mapping,drop_first=False)
dat_mapping.columns = pd.Series(dat_mapping.columns.values).str.split('_',1,True)[1]
dat_mapping = dat_mapping.rename_axis('idx').reset_index()
dat_mapping = dat_mapping.melt('idx',None,'cn','y')
dat_mapping = dat_mapping.groupby(['idx','cn'])['y'].sum().reset_index()
dat_mapping = dat_mapping.pivot('idx','cn','y')
dat_mapping = pd.concat(objs=[enc.df_score,dat_mapping],axis=1)
dat_mapping.to_csv(os.path.join(dir_output, 'dat_mapping.csv'),index=False)
dat_mapping = pd.read_csv(os.path.join(dir_output, 'dat_mapping.csv'))
Xbin = dat_mapping.drop(columns=['idx','n_word','weight'])
cn_drop = Xbin.columns[0]
print('Dropping column - %s' % cn_drop)
Xbin = Xbin.drop(columns=cn_drop)
yval = dat_mapping['weight'].values

mdl_ols = linreg(inference=True)
mdl_ols.fit(X=Xbin.values, y=yval)
dat_ols = pd.DataFrame({'bhat':mdl_ols.bhat,'se':mdl_ols.se}).assign(z=lambda x: x['bhat']/x['se'])
dat_ols.insert(0,'cn',['Intercept'] + list(Xbin.columns))
dat_ols = dat_ols.assign(pval=lambda x: 2*stats.norm.cdf(-np.abs(x['z'])))
dat_ols = dat_ols.assign(bonfer=lambda x: np.minimum(len(x)*x['pval'], 1))
dat_ols =  dat_ols.assign(is_sig=lambda x: x['bonfer'] < 0.05)
dat_ols = dat_ols.query('cn!="Intercept"').assign(cn=lambda x: cat_reorder(x['cn'], x['bhat']))

gtit = 'Linear regression on pair to weight\nDropped column %s' % cn_drop
gg_ols = (pn.ggplot(dat_ols, pn.aes(x='cn',y='bhat',color='is_sig')) +
          pn.theme_bw() + pn.geom_point() + pn.ggtitle(gtit) + 
          pn.geom_hline(yintercept=0,linetype='--') +
          pn.labs(y='Coefficient',x='Cipher pairing') +
          pn.scale_color_discrete(name='Statistically significant (Bonferroni correction)') +
          pn.theme(legend_position=(0.50,0.70),axis_text_x=pn.element_text(angle=90,size=10)))
gg_ols.save(os.path.join(dir_figures,'gg_ols.png'), width=12, height=5)


# (iii) Look into the distribution
pair_range = dat_mapping.drop(columns='idx').melt(['n_word','weight'],None,'pair')
pair_range = pair_range.query('value==1').drop(columns='value')
pair_ord = pair_range.groupby('pair')['n_word'].mean().sort_values(ascending=False).index.to_list()
pair_range['pair'] = pd.Categorical(pair_range['pair'], pair_ord)

gg_range = (pn.ggplot(pair_range, pn.aes(x='pair',y='n_word')) + 
    pn.theme_bw() + pn.geom_boxplot() + 
    pn.labs(y='# of words',x='Cipher pairing') + 
    pn.theme(axis_text_x=pn.element_text(angle=90,size=10)))
gg_range.save(os.path.join(dir_figures,'gg_range.png'), width=12, height=5)


# (iv) Look for hello world
enc3 = encipherer(df_merge, 'word')
enc3.set_letters(letters='etoaisnrlhdw')
n_max = enc3.idx_max['n_encipher']
holder = np.zeros(n_max, dtype=int)
for j in range(1,n_max+1):
    if (j + 1) % 50 == 0:
        print(j+1)
    enc3.set_encipher(idx_pairing=j)
    enc3.get_corpus()
    n_w = enc3.df_encipher['word'].isin(['hello','world']).sum()
    holder[j-1] = n_w
df_hw = pd.DataFrame({'idx':range(1,n_max+1),'n':holder})
df_hw = df_hw.sort_values('n',ascending=False).reset_index(drop=True)
holder = []
for i in df_hw.query('n>0').idx:
    enc3.set_encipher(idx_pairing=i)
    enc3.get_corpus()
    qq = enc3.df_encipher[enc3.df_encipher['word'].isin(['hello','world'])]
    enc3.df_encipher[['word','mirror']].sort_values('word')
    holder.append(qq[['word','mirror']])
    # word = qq['word'].values[0]
    # mirror = qq['mirror'].values[0]
    # print('word = %s, mirror = %s, hash: %s' % (word, mirror, enc3.str_pairing))
zz = pd.concat(holder).drop_duplicates().sort_values('word').reset_index(drop=True)

