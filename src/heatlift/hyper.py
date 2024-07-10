import numpy as np
import itertools as it
from math import comb, factorial
from functools import cache, reduce
from operator import or_
from scipy.sparse import sparray
from collections import defaultdict, Counter
# from scipy.special import comb
from array import array
from scipy.sparse import coo_array

def normalize_hg(H: list):
  """Normalizes a set of hyperedges to a canonical form"""
  V = np.fromiter(reduce(or_, map(set, H)), dtype=int)
  H = [np.unique(np.searchsorted(V, np.sort(he).astype(int))) for he in H]
  return H

## From: https://stackoverflow.com/questions/42138681/faster-numpy-solution-instead-of-itertools-combinations
@cache
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

def downward_closure(H: list, d: int = 1, coeffs: bool = False):
  """Constructs a simplicial complex from a hypergraph by taking its downward closure.
  
  Parameters: 
    H = list of hyperedges / subsets of a set
    d = simplex dimension to extract
    coeffs = whether to extract the membership coefficients 

  Returns: 
    list of the maximal d-simplices in the downward closure of H. 
  """
  assert isinstance(d, int), "simplex dimension must be integral"
  H = normalize_hg(H)
  if not coeffs:
    S = set() # about 15x faster than sortedset 
    for he in H:
      d_simplices = map(tuple, it.combinations(he, d+1)) 
      S.update(d_simplices)
    S = np.array(list(S), dtype=(int, (d+1,)))
    S.sort(axis=1)
    # S = S[np.lexsort(np.rot90(S))]
    return S
  else:
    from scipy.special import comb
    from hirola import HashTable

    ## Extract the lengths of the hyperedges and how many d-simplices we may need
    H_sizes = np.array([len(he) for he in H])
    MAX_HT_SIZE = int(np.sum(comb(H_sizes, d+1)))
    
    ## Allocate the two output containers
    S = HashTable(int(MAX_HT_SIZE * 1.20) + 8, dtype=(int,d+1))
    card_memberships = [array('I') for _ in range(np.max(H_sizes)+1)]
    for he in (he for he in H if len(he) > d):
      d_simplices = he[_combs(len(he), d+1)].T
      s_keys = S.add(d_simplices)
      card_memberships[len(he)-1].extend(s_keys.flatten())

    ## Construct the coauthorship coefficients
    from collections import Counter
    I, J, X = array('I'), array('I'), array('I')
    for j, members in enumerate(card_memberships):
      cc = Counter(members)
      I.extend(cc.keys())
      J.extend(np.full(len(cc), j))
      X.extend(cc.values())
    coeffs = coo_array((X, (I,J)), shape=(len(S), len(card_memberships)))
    coeffs.eliminate_zeros()
    return S.keys.reshape(len(S.keys), d+1), coeffs 

def incidence_to_edges(I: np.ndarray) -> list:
  """Converts an incidence matrix to a list of hyper edges"""
  ## If a list of hyperedges is requested instead
  I,J = np.unravel_index(np.flatnonzero(I), shape=I.shape, order='C')
  hyperedges = np.split(J, np.cumsum(np.unique(I, return_counts=True)[1])[:-1])
  return hyperedges 

def edges_to_incidence(H: list) -> sparray:
  I, J, X = array('I'), array('I'), array('f')
  for i, he in enumerate(H):
    J.extend(he)
    I.extend(np.full(len(he), i))
    X.extend(np.ones(len(he)))
  H = coo_array((X, (I,J)), shape=(len(H), np.max(J) + 1)).T
  return H

def top_weights(simplices: np.ndarray, coeffs: sparray):
  """Computes topological weights from higher-order interaction data."""
  assert isinstance(coeffs, sparray), "Coefficients must be sparse matrix"
  assert coeffs.shape[0] == len(simplices), "Invalid shape; must have a set of coefficients for each simplex"
  simplices = np.atleast_2d(simplices)
  n, N = simplices.shape
  c, d = 1.0 / factorial(N-1), N-1
  _coeff_weights = c * np.array([p / comb(a, d) for p, a in zip(coeffs.data, coeffs.col)])
  top_weights = np.zeros(coeffs.shape[0])
  np.add.at(top_weights, coeffs.row, _coeff_weights)
  return Counter(dict(zip(map(tuple, simplices), top_weights)))

def vertex_counts(H: list) -> np.ndarray:
  """Returns the number of times a """
  N = np.max([np.max(he) for he in normalize_hg(H)])+1
  v_counts = np.zeros(N)
  for he in normalize_hg(H):
    v_counts[he] += 1
  return v_counts


# def edgelist_to_adjacency(edges: np.ndarray, weights: np.ndarray = None, n: int = None):
#   weights = np.asarray(weights) if weights is not None else np.ones(len(edges))
#   assert len(weights) == len(edges), "Invalid weights given; must have one for each edge."
#   n = np.max(edges) if 

