import numpy as np 
import scipy as sp 
from typing import Union
from scipy.sparse.linalg import eigsh, LinearOperator
from numpy.typing import ArrayLike


def time_bounds(L, method: str, interval: tuple = (0.0, 1.0), gap: float = 1e-6, radius: float = 1.0):
  """Returns lower and upper bounds on the time parameter of the heat kernel, based on various heuristics.
  
  The heat kernel (+ its associated invariants) are heavily dependent how the time domain is discretized; 
  this function returns a "reasonable" time interval within which significant diffusion occurs, where 
  the notion of significance is based on the supplied heuristic. 

  Heuristics which infer a suitable time interval [t_min, t_max] include: 
    absolute = returns the interval based on the what can be captured by machine precision.
    bounded = uses gerschgorin-bound on the extremal eigenvalues
    effective = uses supplied extremal eigenvalues (gap + radius)
    simple = the heuristic recommended by the HKS paper. 
  
  The heuristics 'effective' and 'simple' depend on the supplied 'gap' and 'radius' parameters. The 'bounded' 
  approach estimates bounds on these values using O(n) matvecs; the 'absolute' only depends on the dtype of L. 
  """
  machine_eps = np.finfo(L.dtype).eps
  if method == "absolute":
    t_max = -np.log(machine_eps)/machine_eps 
    t_min = 1.0 # since log(0) will be -inf, though this could be go down to 0
    return t_min, t_max
  elif method == "bounded":
    ## use gerschgorin theorem
    min_ew, max_ew = np.inf, 0.0
    cv = np.zeros(L.shape[0])
    for i in range(L.shape[0]):
      cv[i-1], cv[i] = 0, 1
      row = L @ cv
      max_ew = max(row[i] + np.sum(np.abs(np.delete(row, i))), max_ew)
      min_ew = min(row[i] - np.sum(np.abs(np.delete(row, i))), min_ew)
    min_ew = gap if min_ew <= 0 else min_ew 
    l_min = min_ew / np.max(L.diagonal())
    l_max = max_ew / np.min(L.diagonal())
    t_min = 4 * np.log(10) / l_max
    t_max = min(4 * np.log(10) / l_min, -np.log(machine_eps) / min_ew)
    return t_min, t_max
  elif method == "effective":
    l_min, l_max = gap, radius
    t_max = -np.log(machine_eps) / l_min
    t_min = -np.log(machine_eps) / l_max 
    lmi, lmx = np.log(t_min), np.log(t_max)
    t_min = np.exp(1) ** (lmi + interval[0] * (lmx - lmi))
    t_max = np.exp(1) ** (lmi + interval[1] * (lmx - lmi))
    return t_min.item(), t_max.item()
  elif method == "simple":
    l_min, l_max = gap, radius
    t_min = 4 * np.log(10) / l_max
    t_max = 4 * np.log(10) / l_min
    return t_min, t_max
  else: 
    raise ValueError(f"Unknown time bound method '{method}' supplied. Must be one ['absolute', 'gerschgorin', 'effective', 'simple']")