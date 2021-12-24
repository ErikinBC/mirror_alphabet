# Do experiments for nCk
from string import ascii_lowercase
import numpy as np
from scipy.special import comb

lst = list(ascii_lowercase)

def get1(idx, lst, k):
    n = len(lst)
    imax = int(comb(N=n, k=k))
    assert idx >= 1 and idx <= imax
    c, r, j = [], idx, 0
    for s in range(1,k+1):
        cs = j+1
        while r-comb(n-cs,k-s)>0:
            r -= comb(n-cs,k-s)
            cs += 1
        c.append(cs)
        j = cs
    res = [lst[i-1] for i in c]
    return res

get1(idx=1, lst=lst, k=2)
get1(idx=325, lst=lst, k=2)

def get2(idx, lst, k):
    n = len(lst)
    imax = int(comb(N=n, k=k))
    assert idx >= 1 and idx <= imax
    n_k = n - 1
    # Get the interval boundaries
    irange = [0] + [i for i in range(n_k,0,-1)]
    intervals = np.cumsum(irange)
    bins = np.searchsorted(intervals, idx, side='left')
    let1 = lst[bins-1]
    # Second letter
    spread = intervals[bins] - intervals[bins-1]
    within = intervals[bins] - (idx-1)
    idx2 = spread - within
    let2 = lst[bins:][idx2]
    res = [let1, let2]
    return res

get2(idx=1, lst=lst, k=2)
get2(idx=325, lst=lst, k=2)

# THIS NEEDS TO BE GENERALIZED, WE NEED A FOR LOOK OVER K TO DETERMINE HOW MANY POSITIONS
#   1 WILL GET, THEN 2 WILL GET. WHEN K=2, 1 = 8, 2= 7, ... ETC
#   WHEN K=3, 1=(7)+(6)+..+1=28, 2=6+5+..+1=21, 
#   WHEN K=4, 1=[(6)+(5)+..+(1)]+[(5)+(4)+..+(1)]+...=

def find_comb(idx, lst, k):
    n = len(lst)
    imax = int(comb(N=n, k=k))
    assert idx >= 1 and idx <= imax
    # Calculate which combination bin we are in
    n_k = n - (k-1)
    holder_j = np.arange(n_k,0,-1)
    intervals = np.cumsum(holder_j*(holder_j+1) / 2,dtype=int)  #np.append([0], )
    bins = np.searchsorted(intervals, idx, side='left')
    # Loop and assign the indices
    lst0 = lst[bins]
    lst1 = lst[bins+1:]
    print(intervals)
    return lst0, lst1

find_comb(idx=1, lst=lst, k=2)


# def get3(idx, lst, k):
idx = 1
lst = list(range(1,10))
k = 2

idx_j = idx
res = np.zeros(k, dtype=int) - 1
for j in range(1, k):
    lst0, lst1 = find_comb(idx=idx_j, lst=lst, k=2)
    res[j-1] = lst0
    idx
    lst1
    res[j]
    






