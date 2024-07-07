# %% Imports
from networkx.algorithms.shortest_paths import weighted
import numpy as np
from heatlift.data import game_of_thrones
from heatlift.hyper import edges_to_incidence

## Load game of thrones hypergraph data
GOT = game_of_thrones()
hyperedges = list(GOT.values())
incidence = edges_to_incidence(hyperedges)

# %% 
from heatlift.hyper import downward_closure, top_weights
simplices, coeffs = downward_closure(hyperedges, 1, coeffs=True)
simplex_weights = top_weights(simplices, coeffs)

# %% Visualize collaboration network 
from bokeh.plotting import show, figure
from bokeh.io import output_notebook
from map2color import map2hex
from fa2_modified.forceatlas2 import ForceAtlas2
from scipy.sparse import diags
import networkx as nx
output_notebook()

L = (incidence @ incidence.T).tocoo()


G = nx.Graph()
G.add_weighted_edges_from(np.c_[simplices, simplex_weights])
A = nx.adjacency_matrix(G)
communities = nx.community.louvain_communities(G, resolution=1.8, seed=1234)
communities = np.array(communities)[np.argsort([len(c) for c in communities])]

pos = np.array(ForceAtlas2(gravity=1.0).forceatlas2(A))
xs = list(np.c_[pos[simplices[:,0]][:,0], pos[simplices[:,1]][:,0]])
ys = list(np.c_[pos[simplices[:,0]][:,1], pos[simplices[:,1]][:,1]])


p = figure(width=300, height=250)
p.multi_line(xs=xs, ys=ys)
p.scatter(*pos.T)




I = edges_to_incidence(hyperedges)
L = I @ I.T
L.diagonal()