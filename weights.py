import numpy as np
from scipy.sparse import sparray, diags
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


def diffusion(A: sparray, timepoints: np.ndarray, v0: np.ndarray = None):
  assert isinstance(A, sparray) and A.shape[0] == A.shape[1], "Adjacency matrix must be square and sparse"
  
  ## Form the graph Laplacian
  deg = (A @ np.ones(A.shape[1])).ravel()
  L = diags(deg) - A

  ## 
  np.linalg.eigh()