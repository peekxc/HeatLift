import json 
import numpy as np 
from scipy.sparse import coo_array
from .hyper import normalize_hg, edges_to_incidence

def read_hgf(file: str):
  with open(file, "r") as file:
    lines = [line.rstrip() for line in file]
  n, m = map(int, lines[0].split(' '))
  return (n,m), [[int(i.split("=")[0]) for i in L.split(" ")] for L in lines[1:]]

# (n,m), hyperedges = read_hgf("data/algebra.hgf")

def game_of_thrones():
  """Loads the Game of Thrones character/scene hypergraph"""
  with open("data/hg_season1.json", "r") as f:
    hg_s1 = json.load(f)
  n_char, n_scene = hg_s1['n'], hg_s1['k']

  ## The encoding matches characters to their scene numbers; to get a hyper graph, we need the dual 
  characters = np.array(eval(hg_s1['v_meta']))
  char_scene = [np.fromiter(map(int, d.keys()), dtype=int) for d in eval(hg_s1['v2he'].replace("true", "True"))]
  char_scene = normalize_hg(char_scene)
  char2scene = dict(zip(np.arange(n_char), char_scene))

  ## Extract columns to get scene/character hyperedges
  H = edges_to_incidence(char_scene).T.tocsc() ## TODO: replace this
  H.sort_indices()
  hyperedges = np.split(H.indices, H.indptr[1:-1])
  scene2char = dict(zip(np.arange(n_scene), hyperedges))
  
  ## Return the characters + both hypergraph edges
  return {
    "characters" : characters, 
    "scene2char" : scene2char, 
    "char2scene" : char2scene
  }


