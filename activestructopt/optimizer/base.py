from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
  @abstractmethod
  def __init__(self) -> None:
    pass

  @abstractmethod
  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, **kwargs):
    pass
