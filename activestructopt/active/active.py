from activestructopt.optimization.rmc.rmc import rmc_ucb
from activestructopt.gnn.ensemble import Ensemble
from activestructopt.dataset.dataset import make_data_splits, update_datasets
import numpy as np

def active_learning(
    optfunc, 
    args, 
    target,
    config, 
    initial_structure, 
    max_forward_calls = 100,
    rmc_iterations = 10000,
    N = 30, 
    k = 5, 
    perturbrmin = 0.0, 
    perturbrmax = 1.0, 
    split = 1/3, 
    device = 'cuda',
    rmcσ = 0.0025,
    σr = 0.1,
    λ = 1.0,
    print_mses = True,
    ):
  structures, ys, datasets, kfolds, test_indices, test_data, test_targets = make_data_splits(
    initial_structure,
    optfunc,
    args,
    config['dataset'],
    N = N,
    k = k,
    perturbrmin = perturbrmin,
    perturbrmax = perturbrmax,
    split = split,
    device = device,
  )
  mses = [np.mean((y - target_rdfs[0]) ** 2) for y in ys]
  if print_mses:
    print(mses)
  for i in range(max_forward_calls - N):
    starting_structure = structures[np.argmin(mses)].copy()
    ensemble = Ensemble(k, config, datasets)
    ensemble.train()
    ensemble.set_scalar_calibration(test_data, test_targets)
    new_structure = rmc_ucb(
      ensemble.predict,
      {},
      target,
      rmcσ,
      starting_structure,
      rmc_iterations,
      σr = σr,
      λ = λ,
    )
    structures.append(new_structure)
    datasets, y = update_datasets(
      datasets,
      new_structure,
      config['dataset'],
      optfunc,
      args,
      device,
    )
    ys.append(y)
    new_mse = np.mean((y - target) ** 2)
    mses.append(new_mse)
    if print_mses:
      print(new_mse)
  return structures, ys, mses, (
      datasets, kfolds, test_indices, test_data, test_targets, ensemble)
