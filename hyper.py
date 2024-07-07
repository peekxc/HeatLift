import numpy as np
import itertools as it
from math import comb
from functools import cache, reduce
from operator import or_
from scipy.sparse import sparray

def normalize_hg(H: list):
  """Normalizes a set of hyperedges to a canonical form"""
  V = np.fromiter(reduce(or_, map(set, H)), dtype=int)
  H = [np.searchsorted(V, np.sort(he).astype(int)) for he in H]
  return H

## From: https://stackoverflow.com/questions/42138681/faster-numpy-solution-instead-of-itertools-combinations
@cache()
def _combs(n: int, k: int) -> np.ndarray:
  if n < k: return np.empty(shape=(0,),dtype=int)
  a = np.ones((k, n-k+1), dtype=int)
  a[0] = np.arange(n-k+1)
  for j in range(1, k):
    reps = (n-k+j) - a[j-1]
    a = np.repeat(a, reps, axis=1)
    ind = np.add.accumulate(reps)
    a[j, ind[:-1]] = 1-reps[1:]
    a[j, 0] = j
    a[j] = np.add.accumulate(a[j])
  return a

def downward_closure(H: list, d: int = 1, vectorize: bool = False):
  """Constructs a simplicial complex from a hypergraph by taking its downward closure.
  
  Parameters: 
    H = list of hyperedges / subsets of a set
    d = simplex dimension to extract
    vectorize = whether to vectorize (+cache) 

  Returns: 
    list of the maximal d-simplices in the downward closure of H. 
  """
  assert isinstance(d, int), "simplex dimension must be integral"
  H = normalize_hg(H)
  if not vectorize:
    S = set() # about 15x faster than sortedset 
    for he in H:
      d_simplices = map(tuple, it.combinations(he, d+1)) 
      S.update(d_simplices)
    return S
  else:
    from hirola import HashTable
    MAX_HT_SIZE = sum(map(lambda x: comb(len(x), d+1), H))
    S = HashTable(int(MAX_HT_SIZE * 1.20), dtype=(int,d+1))
    for he in (he for he in H if len(he) > d):
      d_simplices = he[_combs(len(he), d+1)].T
      S.add(d_simplices)
    return S.keys

def incidence_to_edges(I: np.ndarray) -> list:
  """Converts an incidence matrix to a list of hyper edges"""
  ## If a list of hyperedges is requested instead
  I,J = np.unravel_index(np.flatnonzero(I), shape=I.shape, order='C')
  hyperedges = np.split(J, np.cumsum(np.unique(I, return_counts=True)[1])[:-1])
  return hyperedges 

def edges_to_incidence(H: list) -> sparray:
  raise NotImplementedError("Not implemented yet")