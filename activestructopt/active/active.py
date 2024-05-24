from activestructopt.dataset.dataset import ASODataset
from activestructopt.gnn.ensemble import Ensemble
from activestructopt.optimization.basinhopping.basinhopping import basinhop
from torch.cuda import empty_cache
from gc import collect
from pickle import dump
from os.path import join as pathjoin

class ActiveLearning():
  def __init__(self, simfunc, target, config, initial_structure, 
    optfunc = basinhop, index = -1):
    self.simfunc = simfunc
    self.config = simfunc.setup_config(config)
    self.optfunc = optfunc
    self.index = index

    self.dataset = ASODataset(initial_structure, target, simfunc, 
      config['dataset'], **(config['aso_params']['dataset']))
    
    self.ensemble = Ensemble(self.dataset.k, config)
  
  def optimize(self, print_mismatches = True, save_progress_dir = None):
    active_steps = self.config['aso_params'][
      'max_forward_calls'] - self.dataset.N

    if print_mismatches:
      print(self.dataset.mismatches)

    for i in range(active_steps):
      train_iters = self.config['optim']['max_epochs'] if i == 0 else (
        self.config['aso_params']['train']['finetune_epochs'])
      lr = self.config['optim']['lr'] if i == 0 else self.config[
        'optim']['lr'] / self.config['aso_params']['train']['lr_reduction']

      self.ensemble.train(self.dataset, iterations = train_iters, lr = lr)
      
      self.ensemble.set_scalar_calibration(self.dataset)
      
      new_structure = self.optfunc(self.ensemble, self.dataset, **(
        self.config['aso_params']['opt']))
      
      self.dataset.update(new_structure)

      if print_mismatches:
        print(self.dataset.mismatches[-1])

      collect()
      empty_cache()
      
      if save_progress_dir is not None:
        self.save(pathjoin(save_progress_dir, str(self.index) + "_" + str(
          i) + ".pkl"))

  def save(self, filename, additional_data = {}):
    res = {'index': self.index,
          'target': self.dataset.target,
          'structures': self.dataset.structures,
          'ys': self.dataset.ys,
          'mismatches': self.dataset.mismatches}
    for k, v in additional_data.items():
      res[k] = v
    with open(filename, "wb") as file:
      dump(res, file)
