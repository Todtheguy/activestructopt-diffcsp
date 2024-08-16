from activestructopt.dataset.base import BaseDataset
from activestructopt.simulation.base import BaseSimulation
from activestructopt.sampler.base import BaseSampler
from pymatgen.core.structure import IStructure
from activestructopt.common.registry import registry
import numpy as np
import copy

@registry.register_dataset("SimpleDataset")
class SimpleDataset(BaseDataset):
  def __init__(self, simulation: BaseSimulation, sampler: BaseSampler, 
    initial_structure: IStructure, target, config, seed = 0, 
    **kwargs) -> None:
    np.random.seed(seed)
    self.simfunc = simulation
    self.N = 1
    self.start_N = 1
    self.target = target
    self.structures = [initial_structure.copy()]
    y_promise = copy.deepcopy(simulation)
    y_promise.get(self.structures[0])
    self.ys = [y_promise.resolve()]
    self.mismatches = [simulation.get_mismatch(self.ys[0], target)]

  def update(self, new_structure: IStructure):
    y_promise = copy.deepcopy(self.simfunc) 
    y_promise.get(new_structure)
    y = y_promise.resolve()
    new_mismatch = self.simfunc.get_mismatch(y, self.target)
    y_promise.garbage_collect(new_mismatch <= min(self.mismatches))

    self.structures.append(new_structure)
    self.ys.append(y)
    self.mismatches.append(new_mismatch)
    self.N += 1
    