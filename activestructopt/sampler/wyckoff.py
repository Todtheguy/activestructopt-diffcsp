from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.common.constraints import lj_reject
from pymatgen.core.structure import IStructure
import numpy as np
from collections import Counter
from pyxtal import pyxtal

@registry.register_sampler("Wyckoff")
class Wyckoff(BaseSampler):
  def __init__(self, initial_structure: IStructure, seed = 0) -> None:
    self.rng = np.random.default_rng(seed)
    element_counter = Counter([site.species.elements[
      0].symbol for site in initial_structure.sites])
    self.zs = list(element_counter.keys())
    self.zcounts = list(element_counter.values())

  def sample(self) -> IStructure:
    rejected = True
    while rejected:
      try:
        xtal = pyxtal()
        xtal.from_random(3, np.random.randint(1, 231), self.zs, self.zcounts,
          random_state = self.rng)
        new_structure = xtal.to_pymatgen()
        rejected = lj_reject(new_structure)
      except:
        rejected = True
    return new_structure
