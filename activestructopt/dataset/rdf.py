from scipy.stats import norm
import numpy as np

def get_dist(a, b):
  return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def get_rdf(structure, rs, σ):
  rmax = np.max(rs) + 3 * σ
  dists = []
  for i in range(len(structure)):
    neighbors = structure.get_sites_in_sphere(structure.sites[i].coords, rmax)
    dists.extend(list(filter(lambda x: x > 0, map(lambda n: get_dist(structure.sites[i].coords, n.coords), neighbors))))
  return sum(map(lambda d: norm.pdf(rs, d, σ), dists))