import numpy as np
import itertools as it
from math import factorial
from simplextree import SimplexTree

## Hypergraph to encode as simplicial complex
H = [(0,),(0,1),(1,3),(1,2,3),(0,1,2,3),(0,1,4)] # ,(2,5),(0,2,5), (0,2,4,5)

## Induce simplicial complex
S = SimplexTree(H)


BASE_WEIGHTS = {
  0: {0: 1}, 
  1: {0: 1, 1: 1}, 
  2: {0: 1, 1: 1/2, 2: 1/2},
  3: {0: 1, 1: 1/3, 2: 1/6, 3: 1/6} # for the 3-simplex
}

## Assign base weights
weights = { tuple(s) : 0 for s in S }
dim = lambda s: len(s) - 1
faces = lambda s: it.chain(*[it.combinations(sigma,d) for d in range(1, len(s))])
for sigma in S.maximal():
  weights[sigma] += 1 / factorial(dim(sigma))
  base_weights = BASE_WEIGHTS[dim(sigma)]
  for f in faces(sigma):
    weights[f] += base_weights[dim(f)]

## Assign augmented weights 
# Try just adding 1 for every non-maximal hyper edge?
initial_weights = weights.copy()
for h in H:
  weights[h] += 1

## Infer hyper-edges from weighted complex 
maximal = S.maximal()
HE = []
for s in S:
  ## if not in maximal
  if s not in maximal and weights[s] >= 1:
    weights[s] -= 1
    HE.append(s)
HE.extend(maximal)
print(HE)
print(H)
HE == H