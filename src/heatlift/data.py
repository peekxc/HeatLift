import json 
import numpy as np 
from .hyper import normalize_hg

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

  ## Each 'scene' is a hyperedge; the scenes
  characters = np.array(eval(hg_s1['v_meta']))
  hyperedges = [np.fromiter(map(int, d.keys()), dtype=int) for d in eval(hg_s1['v2he'].replace("true", "True"))]
  hyperedges = normalize_hg(hyperedges)
  char2scene = dict(zip(characters, hyperedges))

  return char2scene


