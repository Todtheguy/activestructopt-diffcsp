from activestructopt.gnn.dataloader import prepare_data
from activestructopt.optimization.shared.constraints import lj_reject
import numpy as np
import copy

class ASODataset:
  def __init__(self, initial_structure, target, simfunc, config, 
                      perturbrmin = 0.1, perturbrmax = 1.0, 
                      N = 100, split = 0.85, k = 5, device = 'cuda', seed = 0):
    np.random.seed(seed)
    self.device = device
    self.config = config
    self.target = target
    self.initial_structure = initial_structure
    self.perturbrmin = perturbrmin
    self.perturbrmax = perturbrmax
    self.N = N
    self.k = k
    self.simfunc = simfunc
    self.structures = [initial_structure.copy(
      ) if i == 0 else self.random_perturbation() for i in range(N)]
    self.structures[13].to(filename=f"./3_13.cif")
    assert False
    y_promises = [copy.deepcopy(simfunc) for _ in self.structures]
    for i, s in enumerate(self.structures):
      y_promises[i].get(s)
    self.ys = [yp.resolve() for yp in y_promises]
    data = [prepare_data(self.structures[i], config, y = self.ys[i]).to(
      self.device) for i in range(N)]
        
    structure_indices = np.random.permutation(np.arange(1, N))
    trainval_indices = structure_indices[:int(np.round(split * N) - 1)]
    trainval_indices = np.append(trainval_indices, [0])
    self.kfolds = np.array_split(trainval_indices, k)
    self.test_indices = structure_indices[int(np.round(split * N) - 1):]
    self.test_data = [data[i] for i in self.test_indices]
    self.test_targets = [self.ys[i] for i in self.test_indices]
    train_indices = [np.concatenate(
      [self.kfolds[j] for j in range(k) if j != i]) for i in range(k)]
    
    self.datasets = [([data[j] for j in train_indices[i]], 
      [data[j] for j in self.kfolds[i]]) for i in range(k)]

    self.mismatches = [simfunc.get_mismatch(y, target) for y in self.ys]

  def random_perturbation(self):
    rejected = True
    while rejected:
      new_structure = self.initial_structure.copy()
      new_structure.perturb(np.random.uniform(
        self.perturbrmin, self.perturbrmax))
      rejected = lj_reject(new_structure)
    return new_structure

  def update(self, new_structure):
    self.structures.append(new_structure)
    y_promise = copy.deepcopy(self.simfunc) 
    y_promise.get(new_structure)
    y = y_promise.resolve()
    new_mismatch = self.simfunc.get_mismatch(y, self.target)
    y_promise.garbage_collect(new_mismatch <= min(self.mismatches))
    new_data = prepare_data(new_structure, self.config, y = y).to(self.device)
    fold = len(self.datasets) - 1
    for i in range(len(self.datasets) - 1):
      if len(self.datasets[i][1]) < len(self.datasets[i + 1][1]):
        fold = i
        break
    self.datasets[fold][1].append(new_data)
    for i in range(len(self.datasets)):
      if fold != i:
        self.datasets[i][0].append(new_data)
    self.ys.append(y)
    self.mismatches.append(new_mismatch)
    self.N += 1
