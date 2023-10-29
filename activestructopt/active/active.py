from activestructopt.optimization.basinhopping.basinhopping import basinhop, old_ucb_loss, old_mse_loss
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
    N = 30, 
    k = 5, 
    perturbrmin = 0.0, 
    perturbrmax = 1.0, 
    split = 1/3, 
    device = 'cuda',
    bh_starts = 100,
    bh_iters_per_start = 100,
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
  mses = [np.mean((y - target) ** 2) for y in ys]
  if print_mses:
    print(mses)
  active_steps = max_forward_calls - N
  for i in range(active_steps):
    starting_structure = structures[np.argmin(mses)].copy()
    ensemble = Ensemble(k, config, datasets)
    ensemble.train()
    ensemble.set_scalar_calibration(test_data, test_targets)
    new_structure = basinhop(ensemble, starting_structure, target, 
      starts = bh_starts, iters_per_start = bh_iters_per_start, 
      method = "SLSQP", 
      loss_fn = old_mse_loss if i == (active_steps - 1) else old_ucb_loss)
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
