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
        self.formula = initial_structure.formula

    def sample(self) -> IStructure:
        new_structure = sample.main()
        return new_structure
