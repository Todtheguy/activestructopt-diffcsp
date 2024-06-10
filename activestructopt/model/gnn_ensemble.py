from activestructopt.common.dataloader import prepare_data
from activestructopt.model.base import BaseModel, Runner, ConfigSetup
from activestructopt.dataset.base import BaseDataset
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import torch
from torch.func import stack_module_state, functional_call, vmap
import copy
from torch_geometric.loader import DataLoader

@registry.register_model("GNNEnsemble")
class GNNEnsemble(BaseModel):
  def __init__(self, config, k = 5, **kwargs):
    self.k = k
    self.config = config
    self.ensemble = [None for _ in range(k)]
    self.scalar = 1.0
    self.device = 'cpu'
  
  def train(self, dataset: BaseDataset, iterations = 500, lr = 0.001, **kwargs):
    self.config['optim']['max_epochs'] = iterations
    self.config['optim']['lr'] = lr
    for i in range(self.k):
      new_runner = Runner()
      new_runner(self.config, ConfigSetup('train'), 
                            dataset.datasets[i][0], dataset.datasets[i][1])
      if self.ensemble[i] is not None:
        new_runner.trainer.model[0].load_state_dict(
          self.ensemble[i].trainer.model[0].state_dict())
      self.ensemble[i] = new_runner
      self.ensemble[i].train()
      self.ensemble[i].trainer.model[0].eval()
      for l in self.ensemble[i].logstream.getvalue().split('\n'):
        if l.startswith('Epoch: '):
          print(l)
      # https://stackoverflow.com/questions/4330812/how-do-i-clear-a-stringio-object
      self.ensemble[i].logstream.seek(0)
      self.ensemble[i].logstream.truncate(0)
      #self.ensemble[i].trainer.model[0] = compile(self.ensemble[i].trainer.model)
    device = next(iter(self.ensemble[0].trainer.model[0].state_dict().values(
      ))).get_device()
    device = 'cpu' if device == -1 else 'cuda:' + str(device)
    self.device = device
    #https://pytorch.org/tutorials/intermediate/ensembling.html
    models = [self.ensemble[i].trainer.model[0] for i in range(self.k)]
    self.params, self.buffers = stack_module_state(models)
    base_model = copy.deepcopy(models[0])
    self.base_model = base_model.to('meta')
    gnn_mae, _, _ = self.set_scalar_calibration(dataset)
    return gnn_mae, [self.ensemble[i].trainer.metrics for i in range(self.k)]

  def predict(self, structure, prepared = False, mask = None, **kwargs):
    def fmodel(params, buffers, x):
      return functional_call(self.base_model, (params, buffers), (x,))['output']
    data = structure if prepared else [prepare_data(
      structure, self.config['dataset']).to(self.device)]
    prediction = vmap(fmodel, in_dims = (0, 0, None))(
      self.params, self.buffers, next(iter(DataLoader(data, batch_size = len(data)))))

    prediction = torch.mean(torch.transpose(torch.stack(torch.split(prediction, 
      len(mask), dim = 1)), 0, 1)[:, :, torch.tensor(mask, dtype = torch.bool), :], 
      dim = 2) # node level masking

    mean = torch.mean(prediction, dim = 0)
    # last term to remove Bessel correction and match numpy behavior
    # https://github.com/pytorch/pytorch/issues/1082
    std = self.scalar * torch.std(prediction, dim = 0) * np.sqrt(
      (self.k - 1) / self.k)

    return torch.stack((mean, std))

  def set_scalar_calibration(self, dataset: BaseDataset):
    self.scalar = 1.0
    with torch.inference_mode():
      test_res = self.predict(dataset.test_data, prepared = True, 
        mask = dataset.simfunc.mask)
    aes = []
    zscores = []
    for i in range(len(dataset.test_targets)):
      target = np.mean(dataset.test_targets[i][np.array(dataset.simfunc.mask)], 
        axis = 0)
      for j in range(len(target)):
        zscores.append((
          test_res[0][i][j].item() - target[j]) / test_res[1][i][j].item())
        aes.append(np.abs(test_res[0][i][j].item() - target[j]))
    zscores = np.sort(zscores)
    normdist = norm()
    f = lambda x: np.trapz(np.abs(np.cumsum(np.ones(len(zscores))) / len(
      zscores) - normdist.cdf(zscores / x[0])), normdist.cdf(zscores / x[0]))
    self.scalar = minimize(f, [1.0]).x[0]
    return np.mean(aes), normdist.cdf(np.sort(zscores) / 
      self.scalar), np.cumsum(np.ones(len(zscores))) / len(zscores)
