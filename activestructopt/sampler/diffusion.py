from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.common.constraints import lj_reject
from pymatgen.core.structure import IStructure, Lattice
import numpy as np
from diffcsp import sample
from types import SimpleNamespace

@registry.register_sampler("Diffusion")
class Diffusion(BaseSampler):
    def __init__(self, initial_structure: IStructure) -> None:
    self.model_path = '/home/hice1/adaftardar3/DiffCSP-Forked'
    self.save_path = '/home/hice1/adaftardar3/DiffCSP-Forked'
    self.forumla = initial_structure.formula
    self.num_evals = 10
    self.batch_size = 500
    self.step_lr = 1e-5

  def sample(self) -> IStructure:
    data = SimpleNamespace(**{'model_path': self.model_path, 'save_path': self.save_path, 'formula': self.formula, 'num_evals': self.num_evals, 'batch_size': self.batch_size, 'step_lr': self.step_lr})
    new_structure = sample.main(data)
    return new_structure
