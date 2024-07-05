from more_itertools import factor
import numpy as np 
import scipy as sp
from simplextree import SimplexTree
from collections import Counter

## Input: non-negative bare affinity weights for each simplex
## Output: positive topological weights for each simplex
## For collaboration networks, the bare affinity weights are inferred by the number papers co-authored.
def top_weights(authorships: list, aff_weights: np.ndarray = None):
  authorships = list(map(tuple, authorships))
  paper_counts = Counter(authorships)

  st = SimplexTree(authorships)
  d = st.dim()

  ## Initialize topological weights with affinity weights 
  weights = { s : 0 for s in st }
  weights |= dict(zip(authorships, aff_weights)) ## start with affinity weights

  ## 
  for e in st.faces(1):
    weights[e] += np.sum([paper_counts[c] / (len(c) - 1) for c in st.cofaces(e) if c != e])

  ## Final step: the weights of the vertices should match the number their publications
  for i,j in st.edges:
    weights[(i,)] += weights[(i,j)]
    weights[(j,)] += weights[(i,j)]

  ## If validate: 
  # for team in filter(lambda a: len(a) == (d+1), authorships):
  #   weights[tuple(team)] += 1
  return st, weights

## example 1 
authorships = [(0,1)]
aff_weights = [1]
top_weights(authorships, aff_weights)

## example 2
authorships = [(0,1,2)]
aff_weights = [0.5]
top_weights(authorships, aff_weights)

## exmaple 3
authorships = [(0,1,2),(1,2,3),(1,3)]
aff_weights = [0.5, 0.5, 1]
top_weights(authorships, aff_weights)


node_ids = np.loadtxt("https://raw.githubusercontent.com/DedeBac/WeightedSimplicialComplexes/main/nodelist_ConnComp.txt")
edges_mn = np.loadtxt("https://raw.githubusercontent.com/DedeBac/WeightedSimplicialComplexes/main/edges_matrix_with_Mn_ConnComp.csv", skiprows=1, delimiter=",")
triangles_mn = np.loadtxt("https://raw.githubusercontent.com/DedeBac/WeightedSimplicialComplexes/main/tri_matrix_with_Mn_ConnComp.csv", skiprows=1, delimiter=",")

## Remap all ids to [0,1,...,n-1]
node_ids = np.unique(node_ids)
N = np.arange(len(node_ids))
E = np.searchsorted(node_ids, edges_mn[:,:2])
T = np.searchsorted(node_ids, triangles_mn[:,:3])

# test_tri = np.searchsorted(node_ids, [14060612200,56909576000,57223418733])

E_author_card = sp.sparse.coo_array(edges_mn[:,2:])
T_author_card = sp.sparse.coo_array(triangles_mn[:,3:])

from math import factorial, comb
# from scipy.special import factorial, comb
d = 2
c = 1.0 / factorial(d)
T_weights = np.zeros(len(T))
for i in range(len(T)):
  author_mask = T_author_card.row == i
  n_papers = T_author_card.data[author_mask]
  n_coauthors = T_author_card.col[author_mask] + (d+1)
  T_weights[i] = c * np.sum([(p / comb(a, d)) for p, a in zip(n_papers, n_coauthors)])

T_weights_true = np.loadtxt("https://raw.githubusercontent.com/DedeBac/WeightedSimplicialComplexes/main/trilist_weighted.txt", delimiter=",")
# np.searchsorted(node_ids, T_weights_true[:,:3])
assert np.allclose(T_weights, T_weights_true[:,3])


## Triangle weights (from eq. 65, though only restricts to the 2-skeleton)
c, d = 1.0 / factorial(2), 2
T_weights = np.zeros(T_author_card.shape[0])
_T_weights = c * np.array([p / comb(a+(d+1), d) for p, a in zip(T_author_card.data, T_author_card.col)])
np.add.at(T_weights, T_author_card.row, _T_weights)
assert np.allclose(T_weights, T_weights_true[:,3])

## Edge weights (from eq. 65, though only restricts to the 1-skeleton)
c, d = 1.0 / factorial(1), 1 
E_weights = np.zeros(len(E))
_E_weights = c * np.array([p / comb(a+(d+1), d) for p, a in zip(E_author_card.data, E_author_card.col)])
np.add.at(E_weights, E_author_card.row, _E_weights)
E_weights_true = np.loadtxt("https://raw.githubusercontent.com/DedeBac/WeightedSimplicialComplexes/main/edgelist_weighted.txt", delimiter=",")
assert np.allclose(E_weights_true[:,2], E_weights)

## The wedge and triangle weights are correct, but adding edge weights per node does not add correctly 
E.sort(axis=1)
EW_map = dict(zip(map(tuple, E), E_weights))
node_weights = np.array([np.sum([EW_map[tuple(e)] for e in st.cofaces([n]) if len(e) == 2]) for n in st.vertices])

## In no case is the sum of the weights of higher faces equal to the number of papers of the node
# T.sort(axis=1)
# TW_map = dict(zip(map(tuple, T), T_weights))
# W_map = VW_map | EW_map | TW_map
# np.array([np.sum([W_map[tuple(s)] for s in st.cofaces([n])]) for n in st.vertices])

# np.searchsorted(node_ids, E_weights_true[:,:2])

st = SimplexTree([[n] for n in N])
st.insert(E)
st.insert(T)

e1 = np.searchsorted(node_ids, [57194375466,57211365771])
e1

## Should just be: w[I] += J / * comb(D + (d+1), d)
# 0.5 * n_coauthors/comb(n_coauthors, 2) 0.05263157894736842
# (1/2) * (1/comb(19,2)) # 0.0029239766081871343
# (1/2) * (1/comb(20,2)) # 0.002631578947368421

# len(np.unique(np.sort(T, axis=1), axis=1))
# ground truth: 0.00263157894736842


st, weights = top_weights(T, T_weights)

e_weights = np.array([w for s, w in weights.items() if len(s) == 2])
Counter(e_weights)

e1 = np.searchsorted(node_ids, [57194375466,57211365771])
# 268, 323 => 0.2
st.cofaces([268,323])


def read_hgf(file: str):
  with open(file, "r") as file:
    lines = [line.rstrip() for line in file]
  n, m = map(int, lines[0].split(' '))
  return (n,m), [[int(i.split("=")[0]) for i in L.split(" ")] for L in lines[1:]]

(n,m), hyperedges = read_hgf("data/algebra.hgf")

from collections import Counter
## Stores, for each hyper-edge h, a dict (k,v) representing h participating in v papers having k total authors 
collab_sizes = []
for i, he in enumerate(hyperedges):
  C1 = [len(h) for h in hyperedges[:i] if np.all(np.isin(he, h))]
  C2 = [len(h) for h in hyperedges[(i+1):] if np.all(np.isin(he, h))]
  collab_sizes.append(dict(Counter(C1 + C2)))
  print(i)

from math import comb
np.sum([comb(len(h),2) for h in hyperedges])
np.sum([comb(len(h),3) for h in hyperedges])
hyperedges[0]
