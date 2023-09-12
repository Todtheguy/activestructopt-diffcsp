from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data
import activestructopt.gnn.dataloader
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class Runner:
  def __init__(self):
    self.config = None

  def __call__(self, config, args):
    with new_trainer_context(args=args, config=config) as ctx:
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
  def __init__(self, run_mode, config_path):
      self.run_mode = run_mode
      self.config_path = config_path
      self.seed = None
      self.submit = None

class Ensemble:
  def __init__(self, k, config, datafolder):
    self.k = k
    self.config = config
    self.datafolder = datafolder
    self.ensemble = [Runner() for _ in range(k)]
    self.scalar = 1.0
  
  def train(self):
    for i in range(self.k):
      self.config["dataset"]["src"]["train"] = self.datafolder + (
        "/train_k" + str(i) + ".json")
      self.config["dataset"]["src"]["val"] = self.datafolder + (
        "/val_k" + str(i) + ".json")
      self.config["dataset"]["src"]["test"] = self.datafolder + (
        "/test_data.json")
      process_data(self.config["dataset"])
      self.ensemble[i](self.config, ConfigSetup('train', ''))
      self.ensemble[i].trainer.model.eval()

  def predict(self, structure):
    ensemble_results = []
    data = activestructopt.gnn.dataloader.prepare_data(
      structure, device = 'cuda')
    for i in range(self.k):
      ensemble_results.append(
        self.ensemble[i].trainer.model._forward(
        data).cpu().detach().numpy()[0])
    return np.mean(np.array(ensemble_results), 0), np.std(
      np.array(ensemble_results), 0) * self.scalar

  def set_scalar_calibration(self, test_structures, test_targets):
    self.scalar = 1.0
    test_res = [self.predict(s) for s in test_structures]
    zscores = []
    for i in range(len(test_targets)):
      for j in range(len(test_targets[0])):
        zscores.append((
          test_res[i][0][j] - test_targets[i][j]) / test_res[i][1][j])
    zscores = np.sort(zscores)
    normdist = norm()
    f = lambda x: np.trapz(np.abs(np.cumsum(np.ones(len(zscores))) / len(
      zscores) - normdist.cdf(zscores / x[0])), normdist.cdf(zscores / x[0]))
    self.scalar = minimize(f, [1.0]).x[0]
    return normdist.cdf(np.sort(zscores) / self.scalar), np.cumsum(
      np.ones(len(zscores))) / len(zscores)
