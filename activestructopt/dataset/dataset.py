from torch_geometric.data import InMemoryDataset
from activestructopt.gnn.dataloader import prepare_data
import numpy as np

def make_data_splits(initial_structure, optfunc, args, 
                      perturbrmin = 0.1, perturbrmax = 1.0, 
                      N = 100, split = 0.85, k = 5):
  structures = [initial_structure.copy() for i in range(N)]
  for i in range(N):
    structures[i].perturb(np.random.uniform(perturbrmin, perturbrmax))
  ys = [optfunc(structures[i], **(args)) for i in range(N)]

  structure_indices = np.random.permutation(np.arange(N))
  trainval_indices = structure_indices[:int(np.floor(split * N))]
  kfolds = np.array_split(trainval_indices, k)
  test_indices = structure_indices[int(np.floor(split * N)):]
  test_targets = [ys[i] for i in test_indices]
  
  datasets = [([prepare_data(
      structures[x], y = ys[x]) for x in np.concatenate(
      [kfolds[j] for j in range(k) if j != i])], 
      [prepare_data(
      structures[x], y = ys[x]) for x in kfolds[i]]) for i in range(k)]
  
  return structures, datasets, kfolds, test_indices, test_targets
