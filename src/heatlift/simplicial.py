import numpy as np
import itertools as it
from typing import Generator, Sized
from collections import Counter, defaultdict
from math import comb, factorial

def faces(simplex: tuple, proper: bool = False) -> Generator:
  max_dim = len(simplex) - int(proper)
  faces_gen = (it.combinations(simplex, d) for d in range(1, max_dim+1))
  yield from it.chain(*faces_gen)

def dim(simplex: Sized) -> int:
  return len(simplex) - 1

def base_map(dim: int) -> dict:
  """Reciprocal of n! / (n - k)! """
  return Counter({d : 1/(factorial(dim) / factorial(dim-k)) for d, k in enumerate(range(dim+1))})

def weighted_simplex(sigma: tuple) -> dict:
  """Constructs a dictionary mapping faces of 'sigma' to *topological weights*.
  
  The resulting simplex->weight mapping obeys the property that every simplex's weight 
  is strictly positive and is identical to the sum of its cofacet weights. Moreover, 
  every vertex weight is equal to the number of times it appears in a maximal face, 
  and descending order of the weights respect the face poset of the simplex.
  
  This relation is preserved under addition non-enclosing simplex->weight mappings.
  """
  weights = defaultdict(float)
  base_weights = base_map(dim(sigma))
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
def cofacet_constraint(S: dict, d: int = None, relation: str = ">=", verbose: bool = False) -> bool:
  from simplextree import SimplexTree
  # assert relation in 
  st = SimplexTree(S.keys())
  relation_holds: bool = True
  for s in st.faces(d):
    s_weight = S[s]
    s_cofacets = [c for c in st.cofaces(s) if len(c) == len(s) + 1]
    c_weight = np.sum([S[c] for c in s_cofacets])
    relation_holds &= eval(f"s_weight {relation} c_weight") or np.isclose(s_weight, c_weight) or len(s_cofacets) == 0
    if verbose and (not relation_holds) and len(s_cofacets) > 0: 
      print(f"simplex {s} weight {s_weight:.3f} !({relation}) cofacet weight {c_weight:.3f}")
  return relation_holds

def coauthorship_constraint(S: dict, v_counts: np.ndarray) -> bool:
  ## Constraint: the sum of edges of the vertices matches the number of times they appear in the hyper edges 
  same_weight = np.array([np.isclose(S[(i,)], v_counts[i]) for i in np.arange(len(v_counts))])
  return np.all(same_weight)

def positivity_constraint(S: dict) -> bool:
  return np.all([v > 0 for v in S.values()])
