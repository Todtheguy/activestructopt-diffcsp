from activestructopt.common.registry import registry, setup_imports
from torch.cuda import empty_cache
from torch import inference_mode
import numpy as np
from gc import collect
from pickle import dump
from os.path import join as pathjoin
import traceback

class ActiveLearning():
  def __init__(self, simfunc, target, config, initial_structure, 
    index = -1, target_structure = None):
    setup_imports()

    self.simfunc = simfunc
    self.config = simfunc.setup_config(config)
    self.index = index

    self.model_errs = []
    self.model_metrics = []
    self.opt_obj_values = []
    self.new_structure_predictions = []
    self.target_structure = target_structure
    if not (target_structure is None):
      self.target_predictions = []

    sampler_cls = registry.get_sampler_class(
      self.config['aso_params']['sampler']['name'])
    self.sampler = sampler_cls(initial_structure, 
      **(self.config['aso_params']['sampler']['args']))

    dataset_cls = registry.get_dataset_class(
      self.config['aso_params']['dataset']['name'])
    self.dataset = dataset_cls(simfunc, self.sampler, initial_structure, target,
      self.config['dataset'], **(self.config['aso_params']['dataset']['args']))

    model_cls = registry.get_model_class(
      self.config['aso_params']['model']['name'])
    self.model = model_cls(self.config, 
      **(self.config['aso_params']['model']['args']))

    self.error = None
  
  def optimize(self, print_mismatches = True, save_progress_dir = None):
    try:
      active_steps = self.config['aso_params'][
        'max_forward_calls'] - self.dataset.N

      if print_mismatches:
        print(self.dataset.mismatches)

      for i in range(active_steps):
        train_profile = self.config['aso_params']['model']['profiles'][
          np.searchsorted(-np.array(
            self.config['aso_params']['model']['switch_profiles']), 
            -(active_steps - i))]
        opt_profile = self.config['aso_params']['optimizer']['profiles'][
          np.searchsorted(-np.array(
            self.config['aso_params']['optimizer']['switch_profiles']), 
            -(active_steps - i))]
        
        model_err, metrics = self.model.train(self.dataset, **(train_profile))
        self.model_errs.append(model_err)
        self.model_metrics.append(metrics)

        if not (self.target_structure is None):
          with inference_mode():
            self.target_predictions.append(self.model.predict(
              self.target_structure, 
              mask = self.dataset.simfunc.mask).cpu().numpy())

        objective_cls = registry.get_objective_class(opt_profile['name'])
        objective = objective_cls(**(opt_profile['args']))

        optimizer_cls = registry.get_optimizer_class(
          self.config['aso_params']['optimizer']['name'])

        new_structure, obj_values = optimizer_cls().run(self.model, 
          self.dataset, objective, self.sampler, 
          **(self.config['aso_params']['optimizer']['args']))
        self.opt_obj_values.append(obj_values)
        
        self.dataset.update(new_structure)
        with inference_mode():
          self.new_structure_predictions.append(self.model.predict(
            new_structure, 
            mask = self.dataset.simfunc.mask).cpu().numpy())

        if print_mismatches:
          print(self.dataset.mismatches[-1])

        collect()
        empty_cache()
        
        if save_progress_dir is not None:
          self.save(pathjoin(save_progress_dir, str(self.index) + "_" + str(
            i) + ".pkl"))
    except Exception as err:
      print(traceback.format_exc())
      print(err)
      self.error = err


  def save(self, filename, additional_data = {}):
    res = {'al': self}
    for k, v in additional_data.items():
      res[k] = v
    with open(filename, "wb") as file:
      dump(res, file)
