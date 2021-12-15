
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
from funs_cipher import enciphered_dict

letters = [l for l in string.ascii_lowercase]

n_letters_seq = np.arange(2,26+1,2).astype(int)
holder = []
for n_letters in n_letters_seq:
    holder.append(enciphered_dict.n_encipher(0,n_letters))
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

###########################
# --- (4) MODEL CLASS --- #

# set the letters/cipher pairing manually
self=enciphered_dict(df_merge, 'word')
self.set_letters(letters=''.join(top12_letters))
self.set_encipher(idx_pairing=1)
self.get_corpus()
self.df_encipher.loc[1]

print(self.letters)
self.set_encipher(pairing='a:b,c:d')
print(self.mat_pairing)


# Set by index
self=enciphered_dict(df_merge, 'word')
self.set_letters(n_letters=26, idx_letters=1)
print(self.letters)
self.set_encipher(idx_pairing=1)
print(self.mat_pairing)
self.get_corpus()
self.df_encipher


self=enciphered_dict(df_merge, 'word')
self.set_letters(n_letters=26, idx_letters=1)
self.set_encipher(idx_pairing=7000000000000)
print(self.mat_pairing)
self.alpha_trans('abcd')




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


