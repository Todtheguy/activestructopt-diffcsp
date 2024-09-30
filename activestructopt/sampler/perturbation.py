from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.common.constraints import lj_reject
from pymatgen.core.structure import IStructure, Lattice
import numpy as np

@registry.register_sampler("Perturbation")
class Perturbation(BaseSampler):
  def __init__(self, initial_structure: IStructure, perturbrmin = 0.1, 
    perturbrmax = 1.0, perturblσ = 0.1) -> None:
    self.initial_structure = initial_structure
    self.perturbrmin = perturbrmin
    self.perturbrmax = perturbrmax
    self.perturblσ = perturblσ

  def sample(self) -> IStructure:
    rejected = True
    while rejected:
      try:
        new_structure = self.initial_structure.copy()
        new_structure.perturb(np.random.uniform(
          self.perturbrmin, self.perturbrmax))
        new_structure.lattice = Lattice(new_structure.lattice.matrix + 
          self.perturblσ * np.random.normal(0, 1, (3, 3)))
        rejected = lj_reject(new_structure)
      except:
        rejected = True
    return new_structure
