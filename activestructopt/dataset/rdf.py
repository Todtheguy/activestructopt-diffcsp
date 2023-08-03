from scipy.stats import norm
import numpy as np

def get_rdf(structure, σ = 0.05, dr = 0.01, max_r = 12.0):
  rmax = max_r + 3 * σ + dr
  rs = np.arange(0.5, rmax, dr)
  normalization = 4 / structure.volume * np.pi 
  normalization *= (len(structure) * rs) ** 2
  rdf = np.zeros(len(rs))
  site_fcoords = np.mod(structure.frac_coords, 1)
  for i in range(len(structure)):
    # modified from pymatgen's get_site_in_spheres
    for p in structure._lattice.get_points_in_sphere(site_fcoords, 
        structure.sites[i].coords, rmax):
      if 0.5 <= p[1]:
        rdf[int((p[1] - 0.5) / dr)] += 1
  
  return np.convolve(np.array(rdf) / normalization, 
                     norm.pdf(np.arange(-3 * σ, 3 * σ + dr, dr), 0.0, σ), 
                     mode="same")[0:(len(rs) - int((3 * σ) / dr) - 1)]