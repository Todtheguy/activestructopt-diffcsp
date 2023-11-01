from activestructopt.gnn.dataloader import prepare_data
from activestructopt.optimization.basinhopping.basinhopping import get_z, lj_rmins
import numpy as np

def reject(structure):
  for i in range(len(structure)):
    for j in range(i + 1, len(structure)):
      if structure.sites[i].distance(
          structure.sites[j]) < lj_rmins[get_z(
          structure.sites[i])][get_z(structure.sites[j])]:
        return True
  return False

def make_data_splits(initial_structure, optfunc, args, config, 
                      perturbrmin = 0.1, perturbrmax = 1.0, 
                      N = 100, split = 0.85, k = 5, device = 'cuda'):
  structures = [initial_structure.copy() for _ in range(N)]
  for i in range(1, N):
    got_structure = False
    while not got_structure:
      new_structure = initial_structure.copy()
      new_structure.perturb(np.random.uniform(perturbrmin, perturbrmax))
      got_structure = not reject(new_structure)
      structures[i] = new_structure
      
  structure_indices = np.random.permutation(np.arange(1, N))
  trainval_indices = structure_indices[:int(np.floor(split * N) - 1)]
  trainval_indices = np.append(trainval_indices, [0])
  kfolds = np.array_split(trainval_indices, k)
  test_indices = structure_indices[int(np.floor(split * N) - 1):]
  test_data = [data[i] for i in test_indices]
  test_targets = [ys[i] for i in test_indices]
  train_indices = [np.concatenate(
      [kfolds[j] for j in range(k) if j != i]) for i in range(k)]
  
  datasets = [([data[j] for j in train_indices[i]], 
      [data[j] for j in kfolds[i]]) for i in range(k)]
  
  return structures, ys, datasets, kfolds, test_indices, test_data, test_targets

def update_datasets(datasets, new_structure, config, optfunc, args, device):
  y = optfunc(new_structure, **(args))
  new_data = prepare_data(new_structure, config, y = y).to(device)
  fold = len(datasets) - 1
  for i in range(len(datasets) - 1):
    if len(datasets[i][1]) < len(datasets[i + 1][1]):
      fold = i
      break
  datasets[fold][1].append(new_data)
  for i in range(len(datasets)):
    if fold != i:
      datasets[i][0].append(new_data)
  return datasets, y
