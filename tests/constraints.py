import numpy as np
from collections import Counter
from heatlift.simplicial import weighted_simplex, unit_simplex, positivity_constraint, cofacet_constraint, coauthorship_constraint
from heatlift.hyper import vertex_counts, top_weights, downward_closure

def test_constraints():

  ## Simple hypergraph
  H = [(0,1,2), (1,2,3)]
  weights = sum(map(weighted_simplex, H), Counter())

  ## Preserve all the constraints
  assert positivity_constraint(weights)
  assert coauthorship_constraint(weights, vertex_counts(H))
  assert cofacet_constraint(weights)

  ## Respects the vertex/edge constraint, positivity, and coauthorship constraints
  ## Breaks the cofacet constraint
  H = [(0,1,2), (1,2,3), (1,2)]
  weights = sum(map(weighted_simplex, H), Counter())
  assert positivity_constraint(weights)
  assert coauthorship_constraint(weights, vertex_counts(H))
  assert cofacet_constraint(weights, relation=">=")
  assert not cofacet_constraint(weights, relation="==")

  ## Test (trivial) reconstruction with known maximal simplices
  weights = sum(map(weighted_simplex, H[:2]), Counter())
  weights += unit_simplex(H[2])
  H_recon = weights - sum(map(weighted_simplex, H[:2]), Counter())
  assert H_recon == unit_simplex(H[2])

def test_basic_hg():
  H = [(0,),(0,1),(1,3),(1,2,3),(0,1,2,3),(0,1,4),(0,1,3),(2,5),(0,2,5),(0,2,4,5)]
  sc_lift = sum(map(weighted_simplex, H), Counter())
  assert positivity_constraint(sc_lift)
  assert coauthorship_constraint(sc_lift, vertex_counts(H))
  assert cofacet_constraint(sc_lift, d=0, relation=">=")
  assert cofacet_constraint(sc_lift, relation=">=")
  assert not cofacet_constraint(sc_lift, relation="==")

def test_downward_closure():
  from heatlift.hyper import top_weights
  H = [(0,),(0,1),(1,3),(1,2,3),(0,1,2,3),(0,1,4),(0,1,3),(2,5),(0,2,5),(0,2,4,5)]
  sc_lift = sum(map(weighted_simplex, H), Counter())
  for d in range(3):
    d_map = top_weights(*downward_closure(H, d=d, coeffs=True))
    assert np.all([np.isclose(sc_lift[s], w) for s,w in d_map.items()])


