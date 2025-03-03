from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.common.constraints import lj_reject
from pymatgen.core.structure import IStructure, Lattice
import numpy as np
from diffcsp import sample
from types import SimpleNamespace

@registry.register_sampler("Diffusion")
class Diffusion(BaseSampler):
    def __init__(self, initial_structure: IStructure, model_path, save_path, num_evals, batch_size, step_lr) -> None:
        self.formula = initial_structure.formula.replace(' ', "")
        self.model_path = model_path
        self.save_path = save_path
        self.num_evals = num_evals
        self.batch_size = batch_size
        self.step_lr = step_lr
        self.lengths = initial_structure.lattice.abc
        self.angles = initial_structure.lattice.angles

    def sample(self) -> IStructure:
        new_structure = sample.main(self.model_path, self.save_path, self.formula, self.num_evals, self.batch_size, self.step_lr, self.lengths, self.angles)
        return new_structure
