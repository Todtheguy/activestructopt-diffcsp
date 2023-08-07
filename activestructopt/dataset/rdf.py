from scipy.stats import norm
import numpy as np
from pymatgen.optimization.neighbors import find_points_in_spheres

# based heavily on https://github.com/materialsproject/pymatgen/blob/a850e6972b8addc0ecddfefc6394cbb85588f4e4/pymatgen/core/lattice.py#L1412
# to faster get the distances from pymatgen
def get_rdf(structure, σ = 0.05, dr = 0.01, max_r = 12.0):
  rmax = max_r + 3 * σ + dr
  rs = np.arange(0.5, rmax + dr, dr)
  nr = len(rs) - 1
  natoms = len(structure)

  normalization = 4 / structure.volume * np.pi
  normalization *= (natoms * rs[0:-1]) ** 2

  rdf = np.zeros(nr, dtype = int)
  lattice_matrix = np.array(structure.lattice.matrix, dtype=float)
  cart_coords = np.array(structure.cart_coords, dtype=float)

  for i in range(natoms):
    rdf += np.histogram(find_points_in_spheres(
        all_coords = cart_coords,
        center_coords = np.array([cart_coords[i]], dtype=float),
        r = rmax,
        pbc = np.array([1, 1, 1], dtype=int),
        lattice = lattice_matrix,
        tol = 1e-8,
    )[3], rs)[0]

  return np.convolve(rdf / normalization,
                     norm.pdf(np.arange(-3 * σ, 3 * σ + dr, dr), 0.0, σ),
                     mode="same")[0:(nr - int((3 * σ) / dr) - 1)]