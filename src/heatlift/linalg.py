import numpy as np 
import scipy as sp 
from typing import Union
from scipy.sparse.linalg import eigsh, LinearOperator
from numpy.typing import ArrayLike

def heat(x: ArrayLike = None, t: float = 1.0, nonnegative: bool = True, complement: bool = False) -> ArrayLike:
  """Huber loss function."""
  def _heat(x: ArrayLike): 
    x = np.maximum(x, 0.0) if nonnegative else np.abs(x)
    return np.exp(-x * t) if not complement else 1.0 - np.exp(-x * t)
  return _heat if x is None else _heat(x)

def time_bounds(L, A, bound: str, interval: tuple = (0.0, 1.0), dtype = None):
  """Returns lower and upper bounds on the time parameter of the heat kernel, based on various heuristics.
  
  The heat kernel (+ its associated invariants) are heavily dependent how the time domain is discretized; 
  this function returns a 'reasonable' interval of of time to observe the diffusion, based on the supplied heuristic. 

  Heuristics include: 
    absolute = returns [0, t_max], where t_max is the largest t one can expect to have any heat above machine precision.
    laplacian = uses gerschgorin theorem to bound spectral radius and spectral gap
    effective = 
    informative
  """
  # assert len(interval) == 2 and interval[0] >= 0.0 and interval[1] <= 1.0, "If supplied, interval bounds must be in [0,1]"
  if bound == "absolute":
    # assert hasattr(self, "laplacian_")
    # dtype = self.laplacian_.dtype
    dtype = np.float64 if dtype is None else dtype
    t_max = -np.log(np.finfo(dtype).eps)/1e-6 # 1e-6 assumed smallest eigenvalue can be found
    t_min = 1.0 # since log(0) will be -inf, though this could be go down to 0
    return t_min, t_max
  elif bound == "laplacian":
    dtype = L.dtype
    min_ew, max_ew = np.inf, 0.0
    if np.allclose(L.diagonal(), 1.0):
      max_ew = 2.0  ## the normalized laplacian is bounded in [0, p+1]
      min_ew = 1e-6 ## heuristic on spectral gap
    else:
      ## use gerschgorin theorem
      cv = np.zeros(L.shape[0])
      for i in range(L.shape[0]):
        cv[i-1], cv[i] = 0, 1
        row = L @ cv
        max_ew = max(row[i] + np.sum(np.abs(np.delete(row, i))), max_ew)
        min_ew = min(row[i] - np.sum(np.abs(np.delete(row, i))), min_ew)
      min_ew = 1e-6 if min_ew <= 0 else min_ew 
    l_min = min_ew / np.max(self.mass_matrix_.diagonal())
    l_max = max_ew / np.min(self.mass_matrix_.diagonal())
    t_min = 4 * np.log(10) / l_max
    t_max = min(4 * np.log(10) / l_min, -np.log(np.finfo(dtype).eps) / min_ew)
    return t_min, t_max
  elif bound == "effective":
    assert hasattr(self, "eigvals_"), "Must call .fit() first!"
    machine_eps = np.finfo(self.laplacian_.dtype).eps
    l_min, l_max = np.min(self.eigvals_[~np.isclose(self.eigvals_, 0.0)]), np.max(self.eigvals_)
    t_max = -np.log(machine_eps)/l_min
    t_min = -np.log(machine_eps)/(l_max - l_min)
    return t_min, t_max
  elif bound == "informative":
    assert hasattr(self, "eigvals_"), "Must call .fit() first!"
    machine_eps = np.finfo(self.laplacian_.dtype).eps
    # ew = np.sort(self.eigvals_)
    # l_min, l_max = ew[1], ew[-1]
    l_min, l_max = np.min(self.eigvals_[~np.isclose(self.eigvals_, 0.0)]), np.max(self.eigvals_)
    # l_min, l_max = np.quantile(self.eigvals_[1:], interval) ## TODO: revisit, if its a linearly spaced interval use could do themselves
    t_max = -np.log2(machine_eps) / l_min
    t_min = -np.log2(machine_eps) / (l_max - l_min)
    lmi, lmx = np.log2(t_min), np.log2(t_max)
    t_min = 2.0 ** (lmi + interval[0] * (lmx - lmi))
    t_max = 2.0 ** (lmi + interval[1] * (lmx - lmi))
    return t_min, t_max
  elif bound == "heuristic":
    assert hasattr(self, "eigvals_"), "Must call .fit() first!"
    l_min, l_max = np.min(self.eigvals_[~np.isclose(self.eigvals_, 0.0)]), np.max(self.eigvals_) #np.quantile(self.eigvals_, interval)
    t_min = 4 * np.log(10) / l_max
    t_max = 4 * np.log(10) / l_min
    return t_min, t_max
  else: 
    raise ValueError(f"Unknown time bound method '{bound}' supplied. Must be one ['absolute', 'laplacian', 'effective', 'informative', 'heuristic']")
  


def timepoint_heuristic(n: int, L: LinearOperator, A: LinearOperator, locality: tuple = (0, 1), **kwargs):
  """Constructs _n_ positive time points equi-distant in log-space for use in the map exp(-t).
  
  This uses the heuristic from "A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion" to determine 
  adequete time-points for generating a "nice" heat kernel signature, with a tuneable locality parameter. 

  Parameters: 
    n: number of time point to generate 
    L: Laplacian operator used in the the generalized eigenvalue problem.
    A: Mass matrix used in the the generalized eigenvalue problem. 
    locality: tuple indicating how to modify the lower and upper bounds of the time points to adjust for locality. 
  """
  # d = A.diagonal()
  # d_min, d_max = np.min(d), np.max(d)
  # lb_approx = (1.0/d_max)*1e-8
  # tmin_approx = 4 * np.log(10) / (2.0 / d_min)
  # tmax_approx = 4 * np.log(10) / (1e-8 / d_max)
  # TODO: revisit randomized Rayleigh quotient or use known bounds idea
  # XR = np.random.normal(size=(L_sim.shape[0],15), loc=0.0).T
  # np.max([(x.T @ L_sim @ x)/(np.linalg.norm(x)**2) for x in XR])
  lb, ub = eigsh(L, M=A, k=4, which="BE", return_eigenvectors=False, **kwargs)[np.array((1,3))] # checks out
  tmin = 4 * np.log(10) / ub
  tmax = 4 * np.log(10) / lb
  # tdiff = np.abs(tmax-tmin)
  # tmin, tmax = tmin+locality[0]*tdiff, tmax+locality[1]*tdiff
  tmin *= (1.0+locality[0])
  tmax *= locality[1]
  timepoints = np.geomspace(tmin, tmax, n)
  return timepoints 

def logsample(start: float, end: float, num: Union[int, np.ndarray] = 50, endpoint: bool = True, base: int = 2, dtype=None, axis=0):
  """Generate samples on a logarithmic scale within the interval [start, end].

  If 'num' is an integer, the samples are uniformly spaced, matching the behavior of np.logspace. 

  If 'num' is an ndarray, its values are used as relative proportions in log-scale. This can be helpful for procedures 
  seeking to generate e.g. random values that uniformly-sampled in log-scale, i.e. 

  x = logsample(1, 100, np.random.uniform(0,1,size=10))

  Yields 10 random points in the interval [1, 100] that are uniformly-sampled in log-salce.

  Parameters:
    start: The start of the interval.
    end: The end of the interval.
    num: The number of samples to generate, or an array of proportions relative to [start, end].
    endpoint: whether to include end, in the case where num is an integer.
    base: The logarithmic base. Default is 2.
    dtype: passed to np.linspace
    axis: passed to np.linspace 
     
  Returns:
    np.ndarray: An array of logarithmically spaced samples.
  """
  log_start, log_end = np.log(start) / np.log(base),  np.log(end) / np.log(base)
  if isinstance(num, np.ndarray):
    log_samples = log_start + num * np.abs(log_end-log_start)
  else: 
    log_samples = np.linspace(log_start, log_end, num, endpoint=endpoint, dtype=dtype, axis=axis)
  samples = np.power(base, log_samples)
  return samples


# def logspaced_timepoints(n: int, lb: float = 1e-6, ub: float = 2.0) -> np.ndarray:
#   """Constructs _n_ non-negative time points equi-distant in log-space for use in the map exp(-t).
  
#   If an upper-bound for time is known, it may be specified so as to map the interval [0, ub] to the range
#   such that np.exp(-t0) = 1 and np.exp(-tn) = epsilon, which epsilon is the machine epsilon.
#   """
#   ## TODO: revisit the full heuristic! The localized method works so much better
#   # min_t = 13.815510475347063 # 32-bit float 
#   # min_t = 34.53877627071313  # 64-bit floats
#   # if method == "full":
#   #   # tmin = 1e-3 / ub
#   #   # tmax = min_t / max(1e-3, lb)
#   #   tmin = 4 * np.log(10) / 2.0 
#   #   tmax = 4 * np.log(10) / 1e-6
#   #   timepoints = np.geomspace(tmin, tmax, n)
#   # elif method == "local":
#   #   assert lb != 0.0, "Local heuristic require positive lower-bound for spectral gap"
#   tmin = 4 * np.log(10) / ub
#   tmax = 4 * np.log(10) / lb
#   timepoints = np.geomspace(tmin, tmax, n)
#   return timepoints
  # else:
  #   raise ValueError(f"Unknown heuristic method '{method}' passed. Must be one 'local' or 'full'")
  # return timepoints

# def vertex_masses(S: ComplexLike, X: ArrayLike, use_triangles: bool = True) -> np.ndarray:
#   """Computes the cumulative area or 'mass' around every vertex"""
#   # TODO: allow area to be computed via edge lengths?
#   vertex_mass = np.zeros(card(S,0))
#   if use_triangles:
#     from pbsig.shape import triangle_areas
#     areas = triangle_areas(S, X)
#     for t, t_area in zip(faces(S,2), areas):
#       vertex_mass[t] += t_area / 3.0
#   else:
#     for i,j in faces(S,1):
#       edge_weight = np.linalg.norm(X[i] - X[j])
#       vertex_mass[i] += edge_weight / 2.0
#       vertex_mass[j] += edge_weight / 2.0
#   return vertex_mass


