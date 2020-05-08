import numpy as np
import pandas as pd
import string
import os
import shutil
from timeit import timeit
import string
# if not os.path.exists('words.txt'):
#     !wget -q -O words.txt http://www.mieliestronk.com/corncob_lowercase.txt words.txt
#     !wget -q -o words.txt https://raw.githubusercontent.com/dwyl/english-words/master/words.txt

words = pd.read_csv('words.txt',sep='\n',header=None)[0]
words = words[words.notnull()].reset_index(None,True)

letters = [l for l in string.ascii_lowercase]
num_letters = np.arange(1,27)
di_num2let = dict(zip(num_letters, letters))
nletters = 26
# Function to generate paired mapping
def gen_mapping(seed, arr, n):
  np.random.seed(seed)
  return np.random.choice(arr, n, False).reshape([int(n/2),2])
#timeit('gen_mapping(2,num_letters, nletters)',globals=globals(),number=100000)

def alpha_trans(txt, xmap):
  s1 = ''.join(Xmap[:,0])
  s2 = ''.join(Xmap[:,1])
  trans = str.maketrans(s1+s2, s2+s1)
  return txt.str.translate(trans)

Xmap = gen_mapping(35577,letters, nletters)
di_map = dict(zip(Xmap[:,0], Xmap[:,1]))
txt = words.sample(13,random_state=1).copy()
trans = alpha_trans(txt, Xmap)
df_example = pd.concat([pd.DataFrame(Xmap,columns=['from','to']),
                        pd.DataFrame({'original':txt, 'mapped':trans}).reset_index(None,True)],1)
print(df_example)  #[['from','to']]

from time import time

nsim = 80000
tnow = time()

holder = []
for ii in range(nsim):
    if (ii+1) % 10000 == 0:
        tdiff = time() - tnow
        print('Simulation %i of %i (took %i seconds)' % (ii+1, nsim, tdiff))
        tnow = time()
    # Generate a random mapping
    Xmap = gen_mapping(ii, letters, nletters)
    words = words[words.notnull()].reset_index(None,True)
    trans = alpha_trans(words.copy(), Xmap)
    idx = trans.isin(words)
    df = pd.DataFrame({'original':words[idx],'mapped':trans[idx],'seed':ii})
    holder.append(df)
dat_sim = pd.concat(holder).reset_index(None,True)

import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(dat_sim.groupby('seed').size())

mx_idx = dat_sim.groupby('seed').size().sort_values(ascending=False).head(1).index[0]
pd.options.display.max_rows=260
dat_map = dat_sim[dat_sim.seed == mx_idx].reset_index(None,True).assign(ll = lambda x: x.mapped.str.len())
dat_map.sort_values('ll',ascending=False).head(15)

dat_map[['original']].to_csv('fewwords.txt',header=None,index=False,sep='\t')
fewwords = ','.join(dat_map.original).split(',')