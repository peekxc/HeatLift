import numpy as np
import itertools as it
from collections import Counter, defaultdict
from math import comb, factorial

BASE_WEIGHTS = {
  0: {0: 1},                        # 0-simplex
  1: {0: 1, 1: 1},                  # 1-simplex
  2: {0: 1, 1: 1/2, 2: 1/2},        # 2-simplex 
  3: {0: 1, 1: 1/3, 2: 1/6, 3: 1/6} # 3-simplex
}

def weighted_simplex(sigma: tuple) -> dict:
  """Constructs a dictionary mapping faces of 'sigma' to their base topological weights.
  
  The resulting simplex->weight mapping obeys the property that every p-simplex's weight 
  is given by the sum of cofacet weights. Moreover, this relation is preserved under 
  composition with weight maps constructed from other non-face simplices.

  For example, to encode a hypergraph (given its hyperedges) into a weighted simplicial complex with positive weights:
  
  """
  dim = lambda s: len(s) - 1
  faces = lambda s: it.chain(*[it.combinations(sigma,d) for d in range(1, len(s))])
  weights = defaultdict(float)
  weights[tuple(sigma)] = 1 / factorial(dim(sigma))
  base_weights = BASE_WEIGHTS[dim(sigma)]
  for f in faces(sigma):
    weights[f] += base_weights[dim(f)]
  return Counter(weights)

## Ensure the cofacet relations hold; this should hold for all simplices with base weights / "topological weight"
## However this breaks as soon as the affinity weights are added 
def _cofacet_relation(S, weights, verbose: bool = False) -> bool:
  relation_holds: bool = True
  for s in S:
    s_weight = weights[s]
    s_cofacets = cofacets(s)
    c_weight = np.sum([weights[c] for c in s_cofacets])
    relation_holds &= np.isclose(s_weight, c_weight) or len(s_cofacets) == 0
    if verbose and not relation_holds: 
      print(f"simplex {s} weight {s_weight:.3f} != cofacet weight {c_weight:.3f}")
  return relation_holds
# assert cofacet_relation(S, weights), "Cofacet relation doesn't hold"