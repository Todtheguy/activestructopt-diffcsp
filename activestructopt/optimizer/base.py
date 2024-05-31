from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
  @abstractmethod
  def __init__(self) -> None:
    pass

  @abstractmethod
  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, **kwargs):
    pass
