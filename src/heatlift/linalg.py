import numpy as np
import scipy as sp
from typing import Union, Optional
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.sparse import diags, sparray
from scipy.linalg import eigh_tridiagonal
from numpy.typing import ArrayLike


def spectral_bounds(L: sparray) -> tuple:
	"""Estimates the spectral gap and spectral radius of a given Laplacian matrix"""
	radius = eigsh(L, k=1, which="LM", return_eigenvectors=False).item()
	gap = max(1.0 / L.shape[0] ** 4, 0.5 * (2 / np.sum(L.diagonal())) ** 2)
	return {"gap": gap, "radius": radius}


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
		t_max = -np.log(machine_eps) / machine_eps
		t_min = 1.0  # since log(0) will be -inf, though this could be go down to 0
		return t_min, t_max
	elif method == "bounded":
		## use gerschgorin theorem
		min_ew, max_ew = np.inf, 0.0
		cv = np.zeros(L.shape[0])
		for i in range(L.shape[0]):
			cv[i - 1], cv[i] = 0, 1
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
		from scipy.stats import expon

		# l_min, l_max = gap, radius
		# t_min = 4 * np.log(10) / l_max
		# t_max = 4 * np.log(10) / l_min
		b_threshold, e_threshold = expon.ppf(0.001), expon.ppf(0.999)
		t_min = e_threshold / radius
		t_max = b_threshold / gap
		return min([t_min, t_max]), max([t_min, t_max])
	else:
		raise ValueError(
			f"Unknown time bound method '{method}' supplied. Must be one ['absolute', 'gerschgorin', 'effective', 'simple']"
		)


def cmds(D: ArrayLike, d: int = 2, coords: bool = True, pos: bool = True):
	"""Classical Multidimensional Scaling (CMDS).

	Parameters:
		D: _squared_ dense distance matrix.
		d: target dimension of the embedding.
		coords: whether to produce a coordinitization of 'D' or just return the eigen-sets.
		pos: keep only eigenvectors whose eigenvalues are positive. Defaults to True.
	"""
	D = np.asarray(D)
	n = D.shape[0]
	H = np.eye(n) - (1.0 / n) * np.ones(shape=(n, n))  # centering matrix
	evals, evecs = np.linalg.eigh(-0.5 * H @ D @ H)
	evals, evecs = evals[(n - d) : n], evecs[:, (n - d) : n]

	## Compute the coordinates using positive-eigenvalued components only
	if coords:
		w = np.flip(np.maximum(evals, 0.0))
		Y = np.fliplr(evecs) @ np.diag(np.sqrt(w))
		return Y
	else:
		ni = np.setdiff1d(np.arange(d), np.flatnonzero(evals > 0))
		if pos:
			evecs[:, ni], evals[ni] = 1.0, 0.0
		return (np.flip(evals), np.fliplr(evecs))


def pca(X: ArrayLike, d: int = 2, scale: bool = True, coords: bool = True, svd: bool = False) -> np.ndarray:
	"""Principal Component Analysis (PCA).

	Parameters:
		X: (n x D) point cloud / design matrix.
		d: target dimension of the embedding.
		scale: whether to standardize the dimensions
		coords: whether to produce a coordinitization of 'D' or just return the eigen-sets.
		svd: whether to use the svd for the computation instead. Defaults to False.
	"""
	X = np.asarray(X)  # only work with numpy arrays
	mu, Sigma = X.mean(axis=0), np.std(X, axis=0) if scale else 1  # sometimes scale
	X -= np.where(np.isnan(mu), 0.0, mu)  # always center
	X /= np.where(np.isclose(Sigma, 0.0), 1.0, Sigma)  # standardize variance if correlation is more informative

	if not svd:
		Sigma = np.cov(X, rowvar=False)  # covariance == correlation if scale = True
		ew, ev = np.linalg.eigh(Sigma)  # psd => eigh
		ind = ew.argsort()[::-1][:d]  # largest eigenvalues
		return X @ ev[:, ind] if coords else (ew[ind], ev[:, ind])
	else:
		u, s, vt = np.linalg.svd(X)
		ind = s.argsort()[::-1][:d]
		return u[:, ind] @ np.diag(s[ind]) if coords else (u[:, ind], s[ind], vt[ind, :])


def _hks_approx(A: LinearOperator, timepoints, maxiter: int = 200, deg: int = 20):
	# ask_pkg_install("primate")
	from primate.diagonalize import lanczos

	hks = np.zeros((A.shape[0], len(timepoints)))  # heat kernel signature
	hkt = np.zeros(len(timepoints))  # heat kernel trace
	for i in range(maxiter):
		x = np.random.choice([-1, +1], size=A.shape[1])
		x = x / np.linalg.norm(x)
		(a, b), Q = lanczos(A, v0=x, deg=deg, return_basis=True)
		s, Y = eigh_tridiagonal(a, b)  # rayleigh ritz values
		for i, t in enumerate(timepoints):
			hks[:, i] += x * (Q @ Y @ np.diag(np.exp(-t * s)) @ Y.T)[:, 0] / (x * x)
			hkt[i] += np.sum(np.exp(-t * s))
	hks /= maxiter
	# hkt /= maxiter
	return hks, hkt


def heat_kernel_signature(
	L: sparray,
	timepoints: Union[int, np.ndarray],
	approx: bool = False,
	scale: bool = True,
	subset: Optional[list] = None,
	**kwargs,
):
	if not approx:
		ew, ev = np.linalg.eigh(L.todense())
		index_set = np.arange(ev.shape[0]) if subset is None else np.array(subset).astype(int)
		cind_nz = np.flatnonzero(~np.isclose(ew, 0.0, atol=1e-14))
		ev_subset = np.square(ev[np.ix_(index_set, cind_nz)]) if subset is not None else np.square(ev[:, cind_nz])
		hks_matrix = np.array([np.ravel(ev_subset @ np.exp(-t * ew[cind_nz])) for t in timepoints]).T
		hkt = np.array([np.sum(np.exp(-t * ew[cind_nz])) for t in timepoints])
	else:
		hks_matrix, hkt = _hks_approx(L, timepoints, **kwargs)
	if scale:
		ht = np.reciprocal(hkt, where=~np.isclose(hkt, 0.0, atol=1e-14))
		hks_matrix = hks_matrix @ diags(ht)
	return hks_matrix, hkt
