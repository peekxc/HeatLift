from collections import defaultdict
import numpy as np
import itertools as it
from math import factorial
from notebooks.simple import top_weights
from simplextree import SimplexTree
from heatlift.simplicial import weighted_simplex
from collections import Counter

from src.heatlift.hyper import normalize_hg

## Hypergraph to encode as simplicial complex
H = [(0,),(0,1),(1,3),(1,2,3),(0,1,2,3),(0,1,4),(0,1,3),(0,)] # ,(2,5),(0,2,5), (0,2,4,5)

## Induce simplicial complex via downward closure
S = SimplexTree(H)

## Construct weight mapping using maximal simplices
weights = sum(map(weighted_simplex, H), Counter())

max_weights = sum(map(weighted_simplex, SimplexTree(weights.keys()).maximal()), Counter())
sum(map(weighted_simplex, weights.keys()), Counter())

weights = weighted_simplex([0,1,2]) + weighted_simplex([0,1])
weights - weighted_simplex([0,1,2])


# weights - sum(map(weighted_simplex, weights.keys()), Counter())

# weights = sum(map(weighted_simplex, S.maximal()), Counter())
# faces = lambda s: it.chain(*(it.combinations(s,k) for k in range(1, len(s)+1)))
# aff_weights = sum([Counter(faces(s)) for s in H if s not in S.maximal()], Counter())
# weights += aff_weights
# weights += Counter(H)

list(faces((0,1)))

## The hyperedges can be recovered by subtracting weight map of the maximal simplices
HE = weights - sum(map(weighted_simplex, S.maximal()), Counter())
assert list(sorted(HE.keys())) == list(sorted(set(H)))

## Constraint: the sum of edges of the vertices matches the number of times they appear in the hyper edges 
N = np.max([np.max(he) for he in normalize_hg(H)])+1
v_counts = np.zeros(N)
for he in normalize_hg(H):
  v_counts[he] += 1
v_weights = np.array([weights[(i,)] for i in range(N)])
np.allclose(v_weights, v_counts)

## Problem: can't construct the full induced complex; it's too large! 
## Can we instead construct, from the hyperedges alone, a weighted d-skeleton?
## Ideally we want: weight map matches that of the above but restricted to d-simplices 
## This could work if we knew, for each coefficient, whether the corresponding inclusion was maximal
edges, coeffs = downward_closure(H, 1, coeffs=True)
sk_map = defaultdict(float)
for i,s in enumerate(map(tuple, edges)):
  inc_dim = coeffs.col[coeffs.row == i] - 1
  inc_mul = coeffs.data[coeffs.row == i]
  # sk_map[s] += sum([BASE_WEIGHTS[d][1]*m for d,m in zip(inc_dim, inc_mul)])

## Alternatively, we consider the simplicial complex formed by the d-skeleton  
d_faces = lambda s, d: it.chain(*(it.combinations(s, p) for p in range(1, d+1)))
HS = [(0,),(0, 1),(1, 3),(1, 2, 3),(0, 1, 2),(0, 1, 3),(0, 2, 3),(1, 2, 3),(0, 1, 4),(0, 1, 3),(0,)] 
# list(it.chain(*[it.combinations(h, 3) if len(h) > 3 else [h] for h in H ]))

## The goal / weight map to replicate the weight produced by this
S = SimplexTree(HS)
weights = sum(map(weighted_simplex, S.maximal()), Counter())
weights += Counter(HS)
{ k : v for k,v in weights.items() if len(k) == 3 }
# {(0, 1, 2): 1.5,
#  (0, 1, 3): 2.5,
#  (0, 1, 4): 1.5,
#  (0, 2, 3): 1.5,
#  (1, 2, 3): 2.5}

## TODO: identify maximal simplices from hyperedges, and only count those
is_subset = lambda s1, s2: np.all(np.isin(s1, s2)) if len(s1) <= len(s2) else False
H_subset = np.array([is_subset(h1, h2) for h1, h2 in it.product(H, H)]).reshape((len(H), len(H)))
maximal_he = list(it.compress(H, H_subset.sum(axis=1) == 1))



from heatlift.hyper import downward_closure
triangles, coeffs = downward_closure(H, 2, coeffs=True)
coeffs.todense()
# top_weights(*downward_closure(H, 2, coeffs=True))
# { k : v for k,v in weights.items() if len(k) == 3 }

# d_faces = lambda s, d: it.combinations(s, d+1)
d_faces = lambda s, d: it.chain(*(it.combinations(s, p) for p in range(1, d+1)))
weights = sum(map(weighted_simplex, it.chain(*(d_faces(h, 2) for h in H))), Counter())

weights = sum(map(weighted_simplex, S.faces(1)), Counter())
weights += Counter(H)

# weights - sum(map(weighted_simplex, S.faces(1)), Counter())

s = (0,1,2,3)

faces()
# from heatlift.hyper import downward_closure, top_weights
# edges, coeffs = downward_closure(H, 3, coeffs=True)

top_weights(*downward_closure([(0,1,2,3)], 3, coeffs=True))

top_weights(edges, coeffs)

# top_weights(edges, coo_array(np.c_[np.zeros(coeffs.shape[0]), coeffs.todense()]))



## Hypergraph to encode as simplicial complex
H = [(0,),(0,1),(1,3),(1,2,3),(0,1,2,3),(0,1,4)] # ,(2,5),(0,2,5), (0,2,4,5)

# ## Assign base weights
# weights = { tuple(s) : 0 for s in S }
# dim = lambda s: len(s) - 1
# faces = lambda s: it.chain(*[it.combinations(sigma,d) for d in range(1, len(s))])
# for sigma in S.maximal():
#   weights[sigma] += 1 / factorial(dim(sigma))
#   base_weights = BASE_WEIGHTS[dim(sigma)]
#   for f in faces(sigma):
#     weights[f] += base_weights[dim(f)]




# ## Assign augmented weights 
# # Try just adding 1 for every non-maximal hyper edge?
# cofacets = lambda s: [c for c in S.cofaces(s) if dim(c) == (dim(s) + 1)]
# initial_weights = weights.copy()
# for h in H:
#   # weights[h] += 1
#   C = cofacets(h)
#   while len(C) > 0:
#     for c in C:
#       weights[c] += 1/len(C)

# #    
# h_order = np.argsort([-len(h) for h in H])

# ## Infer hyper-edges from weighted complex 
# maximal = S.maximal()
# for k,v in initial_weights.items():
#   weights[k] -= v
# HE = [k for k,v in weights.items() if v > 0]

# assert list(sorted(H)) == list(sorted(HE))
# # for s in S:
# #   ## if not in maximal
# #   if s not in maximal and weights[s] >= 1:
# #     weights[s] -= 1
# #     HE.append(s)
# # HE.extend(maximal)



# print(HE)
# print(H)
# HE == H