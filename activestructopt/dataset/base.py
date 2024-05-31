from abc import ABC, abstractmethod
from pymatgen.core.structure import IStructure
from activestructopt.simulation.base import BaseSimulation

class BaseDataset(ABC):
  @abstractmethod
  def __init__(self, simulation: BaseSimulation, initial_structure: IStructure, 
    target, config, **kwargs):
    pass

  @abstractmethod
  def sample(self):
    pass

  @abstractmethod
  def update(self, new_structure: IStructure):
    pass
