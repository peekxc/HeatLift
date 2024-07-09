import numpy as np
import itertools as it
from typing import Generator, Sized
from collections import Counter, defaultdict
from math import comb, factorial

BASE_WEIGHTS = {
  0: {0: 1},                        # 0-simplex
  1: {0: 1, 1: 1},                  # 1-simplex
  2: {0: 1, 1: 1/2, 2: 1/2},        # 2-simplex 
  3: {0: 1, 1: 1/3, 2: 1/6, 3: 1/6} # 3-simplex
}

def faces(simplex: tuple, proper: bool = False) -> Generator:
  max_dim = len(simplex) - int(proper)
  faces_gen = (it.combinations(simplex, d) for d in range(1, max_dim+1))
  yield from it.chain(*faces_gen)

def dim(simplex: Sized) -> int:
  return len(simplex) - 1

def weighted_simplex(sigma: tuple) -> dict:
  """Constructs a dictionary mapping faces of 'sigma' to their base topological weights.
  
  The resulting simplex->weight mapping obeys the property that every p-simplex's weight 
  is given by the sum of cofacet weights. Moreover, this relation is preserved under 
  composition with weight maps constructed from other non-face simplices.
  """
  weights = defaultdict(float)
  base_weights = BASE_WEIGHTS[dim(sigma)]
  for f in faces(sigma, proper=True):
    weights[f] += base_weights[dim(f)]
  weights[tuple(sigma)] = 1 / factorial(dim(sigma))
  return Counter(weights)

def unit_simplex(sigma: tuple, c: float = 1.0, closure: bool = False) -> dict:
  weights = defaultdict(float)
  if closure: 
    for f in faces(sigma, proper=True):
      weights[f] += c
  weights[tuple(sigma)] = c 
  return Counter(weights)

## Ensure the cofacet relations hold; this should hold for all simplices with base weights / "topological weight"
## However this breaks as soon as the affinity weights are added 
def cofacet_constraint(S: dict, d: int = None, verbose: bool = False) -> bool:
  from simplextree import SimplexTree
  st = SimplexTree(S.keys())
  relation_holds: bool = True
  for s in st.faces(d):
    s_weight = S[s]
    s_cofacets = [c for c in st.cofaces(s) if len(c) == len(s) + 1]
    c_weight = np.sum([S[c] for c in s_cofacets])
    relation_holds &= np.isclose(s_weight, c_weight) or len(s_cofacets) == 0
    if verbose and (not relation_holds) and len(s_cofacets) > 0: 
      print(f"simplex {s} weight {s_weight:.3f} != cofacet weight {c_weight:.3f}")
  return relation_holds

def coauthorship_constraint(S: dict, v_counts: np.ndarray) -> bool:
  ## Constraint: the sum of edges of the vertices matches the number of times they appear in the hyper edges 
  same_weight = np.array([np.isclose(S[(i,)], v_counts[i]) for i in np.arange(len(v_counts))])
  return np.all(same_weight)

def positivity_constraint(S: dict) -> bool:
  return np.all([v > 0 for v in S.values()])

# assert _cofacet_relation(weighted_simplex([0,1,2,3]))
# assert cofacet_relation(S, weights), "Cofacet relation doesn't hold"