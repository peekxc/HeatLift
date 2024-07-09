import numpy as np
from collections import Counter
from heatlift.simplicial import weighted_simplex, unit_simplex, positivity_constraint, cofacet_constraint, coauthorship_constraint
from heatlift.hyper import vertex_counts

def test_constraints():

  ## Simple hypergraph
  H = [(0,1,2), (1,2,3)]
  weights = sum(map(weighted_simplex, H), Counter())

  ## Preserve all the constraints
  assert positivity_constraint(weights)
  assert coauthorship_constraint(weights, vertex_counts(H))
  assert cofacet_constraint(weights)
  assert cofacet_constraint(weights, d=0) # vertex/edge only

  ## Respects the vertex/edge constraint, positivity, and coauthorship constraints
  ## Breaks the cofacet constraint
  H = [(0,1,2), (1,2,3), (1,2)]
  weights = sum(map(weighted_simplex, H), Counter())
  assert positivity_constraint(weights)
  assert coauthorship_constraint(weights, vertex_counts(H))
  assert not cofacet_constraint(weights)
  assert cofacet_constraint(weights, d=0) # vertex/edge only

  ## Test (trivial) reconstruction with known maximal simplices
  weights = sum(map(weighted_simplex, H[:2]), Counter())
  weights += unit_simplex(H[2])
  H_recon = weights - sum(map(weighted_simplex, H[:2]), Counter())
  assert H_recon == unit_simplex(H[2])