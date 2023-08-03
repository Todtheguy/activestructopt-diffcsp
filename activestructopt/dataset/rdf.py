from scipy.stats import norm
import numpy as np

def get_dist(a, b):
  return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def get_rdf(structure, σ = 0.05, dr = 0.01, max_r = 12.0):
  rs = np.arange(0.5, max_r + 5 * σ + dr, dr)
  p = len(structure) / structure.volume
  rmax = np.max(rs) + 3 * σ
  dists = []
  for i in range(len(structure)):
    neighbors = structure.get_sites_in_sphere(structure.sites[i].coords, rmax)
    dists.extend(list(filter(lambda x: x > 0, 
      map(lambda n: get_dist(structure.sites[i].coords, n.coords), neighbors))))
  rdf = []
  dists = np.array(dists)
  for r in rs:
    inds = dists < (r + dr)
    rdf.append(np.sum(inds) / (len(structure) * p * 4 * np.pi * r ** 2))
    dists = dists[np.invert(inds)]
  return np.convolve(rdf, norm.pdf(np.arange(-5 * σ, 5 * σ + dr, dr), 0.0, σ), mode="same")[0:(len(rs) - int(np.floor((5 * σ) / dr)) - 1)]