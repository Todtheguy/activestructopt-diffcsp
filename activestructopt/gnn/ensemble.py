from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data
import activestructopt.gnn.dataloader
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from torch_geometric import compile
import torch
import torch.multiprocessing as mp
from torch.func import stack_module_state, functional_call, vmap
import copy

class Runner:
  def __init__(self):
    self.config = None

  def __call__(self, config, args):
    with new_trainer_context(args = args, config = config) as ctx:
        self.config = ctx.config
        self.task = ctx.task
        self.trainer = ctx.trainer
        self.task.setup(self.trainer)
        self.task.run()

  def checkpoint(self, *args, **kwargs):
    self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
    self.config["checkpoint"] = self.task.chkpt_path
    self.config["timestamp_id"] = self.trainer.timestamp_id

class ConfigSetup:
  def __init__(self, run_mode, train_data, val_data):
      self.run_mode = run_mode
      self.seed = None
      self.submit = None
      self.datasets = {
        'train': train_data, 
        'val': val_data, 
      }

def train_model_func(params):
  model = Runner()
  config, train_data, val_data = params
  model(config, ConfigSetup('train', train_data, val_data))
  model.trainer.model.eval()
  return copy.deepcopy(model)

def predict_model_func(params):
  model, data = params
  return copy.deepcopy(model._forward(data))

class Ensemble:
  def __init__(self, k, config, datasets):
    self.k = k
    self.config = config
    self.datasets = datasets
    self.ensemble = []
    self.scalar = 1.0
    self.device = 'cpu'
    mp.set_start_method('spawn')
  
  def train(self):
    with mp.Pool(5) as p:
      self.ensemble = p.map(train_model_func, zip( 
        [copy.deepcopy(self.config) for _ in range(self.k)],
        [copy.deepcopy(self.datasets[i][0]) for i in range(self.k)],
        [copy.deepcopy(self.datasets[i][1]) for i in range(self.k)]))
    #for i in range(self.k):
    #  self.ensemble[i].trainer.model = compile(self.ensemble[i].trainer.model)
    device = next(iter(self.ensemble[0].trainer.model.state_dict().values(
      ))).get_device()
    device = 'cpu' if device == -1 else 'cuda:' + str(device)
    self.device = device

  def predict(self, structure, prepared = False):
    if not prepared:
      data = activestructopt.gnn.dataloader.prepare_data(
        structure, self.config['dataset']).to(self.device)
    else:
      data = structure

    with mp.Pool(5) as p:
      prediction = torch.stack(p.map(train_model_func, zip( 
        [copy.deepcopy(self.ensemble[i].trainer.model) for i in range(self.k)],
        [copy.deepcopy(data) for i in range(self.k)])))
    print(prediction.size)
    print(prediction)

    mean = torch.mean(prediction, dim = 0)
    # last term to remove Bessel correction and match numpy behavior
    # https://github.com/pytorch/pytorch/issues/1082
    std = self.scalar * torch.std(prediction, dim = 0) * np.sqrt(
      (self.k - 1) / self.k)

    return mean, std

  def set_scalar_calibration(self, test_data, test_targets):
    self.scalar = 1.0
    test_res = [self.predict(s, prepared = True) for s in test_data]
    zscores = []
    for i in range(len(test_targets)):
      for j in range(len(test_targets[0])):
        zscores.append((
          test_res[i][0][j].detach().numpy() - test_targets[i][j]) / test_res[i][1][j].detach().numpy())
    zscores = np.sort(zscores)
    normdist = norm()
    f = lambda x: np.trapz(np.abs(np.cumsum(np.ones(len(zscores))) / len(
      zscores) - normdist.cdf(zscores / x[0])), normdist.cdf(zscores / x[0]))
    self.scalar = minimize(f, [1.0]).x[0]
    return normdist.cdf(np.sort(zscores) / self.scalar), np.cumsum(
      np.ones(len(zscores))) / len(zscores)
