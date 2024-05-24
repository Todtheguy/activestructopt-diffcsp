from activestructopt.optimization.basinhopping.basinhopping import basinhop
from activestructopt.gnn.ensemble import Ensemble
from activestructopt.dataset.dataset import make_data_splits, update_datasets
from activestructopt.optimization.shared.constraints import lj_reject
import numpy as np
import gc
import torch
import pickle
import sys

def active_learning(
    optfunc, 
    target,
    config, 
    initial_structure, 
    max_forward_calls = 100,
    N = 30, 
    k = 5, 
    perturbrmin = 0.0, 
    perturbrmax = 1.0, 
    split = 1/3, 
    device = 'cuda',
    bh_starts = 128,
    bh_iters_per_start = 100,
    bh_lr = 0.01,
    print_mismatches = True,
    save_progress_dir = None,
    λ = 1.0,
    seed = 0,
    finetune_epochs = 500,
    lr_reduction = 1.0,
    ):
  structures, ys, mismatches, datasets, kfolds, test_indices, test_data, test_targets = make_data_splits(
    initial_structure,
    target,
    optfunc,
    config['dataset'],
    N = N,
    k = k,
    perturbrmin = perturbrmin,
    perturbrmax = perturbrmax,
    split = split,
    device = device,
    seed = seed,
  )
  config = optfunc.setup_config(config)
  lr1, lr2 = config['optim']['lr'], config['optim']['lr'] / lr_reduction
  if print_mismatches:
    print(mismatches)
  active_steps = max_forward_calls - N
  ensemble = Ensemble(k, config)
  for i in range(active_steps):
    starting_structures = [initial_structure.copy() for _ in range(bh_starts)]
    for j in range(np.minimum(len(structures), bh_starts)):
      starting_structures[j] = structures[j].copy()
    if len(structures) < bh_starts:
      for j in range(len(structures), bh_starts):
        rejected = True
        while rejected:
          new_structure = initial_structure.copy()
          new_structure.perturb(np.random.uniform(perturbrmin, perturbrmax))
          rejected = lj_reject(new_structure)
        starting_structures[j] = new_structure.copy()

    ensemble.train(datasets, iterations = config['optim'][
      'max_epochs'] if i == 0 else finetune_epochs, lr = lr1 if i == 0 else lr2)
    ensemble.set_scalar_calibration(test_data, test_targets, mask = optfunc.mask)
    new_structure = basinhop(ensemble, starting_structures, target, 
      config['dataset'], niters = bh_iters_per_start, 
      λ = 0.0 if i == (active_steps - 1) else λ, lr = bh_lr,
      mask = optfunc.mask)
    structures.append(new_structure)
    datasets, ys, mismatches = update_datasets(
      datasets,
      new_structure,
      config['dataset'],
      optfunc,
      device,
      ys,
      mismatches,
      target,
    )
    if print_mismatches:
      print(mismatches[-1])
    gc.collect()
    torch.cuda.empty_cache()
    if save_progress_dir is not None:
      res = {'index': sys.argv[1],
            'iter': i,
            'structures': structures,
            'ys': ys,
            'mismatches': mismatches}

      with open(save_progress_dir + "/" + str(sys.argv[1]) + "_" + str(i) + ".pkl", "wb") as file:
          pickle.dump(res, file)

  return structures, ys, mismatches, (
      datasets, kfolds, test_indices, test_data, test_targets, ensemble)
