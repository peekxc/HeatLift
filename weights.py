import numpy as np
from typing import Sequence, Iterable

from combin import rank_to_comb, comb_to_rank

## weights = simplex affinity weights (non-negative). 
def optimal_weights(simplices: Sequence, weights: Iterable = None) -> np.ndarray:
  """

  Larger values indicate the vertices in simplex have an affinity to each other.

  Parameters: 
    weights = bare affinity weights (non-negative).
  Returns: 
    positive topological weights for every simplex in the complex. 
  """
  simplices = list(simplices)
  weights = np.fromiter(weights, dtype=np.float64)
  return weights 

optimal_weights(*d.items())