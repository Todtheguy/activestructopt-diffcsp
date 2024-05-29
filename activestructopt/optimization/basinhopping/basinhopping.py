import torch
import numpy as np
from activestructopt.gnn.dataloader import prepare_data, reprocess_data
from activestructopt.optimization.shared.constraints import lj_rmins, lj_repulsion
from activestructopt.optimization.shared.objectives import ucb_obj

def run_adam(ensemble, target, starting_structures, config, ljrmins,
                    niters = 100, lr = 0.01, mask = None, 
                    obj_func = ucb_obj, obj_args = {'λ': 1.0}, device = 'cpu'):
  nstarts = len(starting_structures)
  natoms = len(starting_structures[0])
  best_obj = torch.tensor([float('inf')], device = device)
  best_x = torch.zeros(3 * natoms, device = device)
  target = torch.tensor(target, device = device)
  data = [prepare_data(s, config, pos_grad = True, device = device, 
    preprocess = False) for s in starting_structures]
  for i in range(nstarts): # process node features
    reprocess_data(data[i], config, device, edges = False)
                      
  optimizer = torch.optim.Adam([d.pos for d in data], lr=lr)
  
  split = int(np.ceil(np.log2(nstarts)))
  orig_split = split

  for i in range(niters):
    predicted = False
    while not predicted:
      try:
        optimizer.zero_grad(set_to_none=True)
        for j in range(nstarts):
          data[j].pos.requires_grad_()
          reprocess_data(data[j], config, device, nodes = False)

        for k in range(2 ** (orig_split - split)):
          starti = k * (2 ** split)
          stopi = min((k + 1) * (2 ** split) - 1, nstarts - 1)
          predictions = ensemble.predict(data[starti:(stopi+1)], 
            prepared = True, mask = mask)

          objs, obj_total = obj_func(predictions, target, device = device, 
            N = stopi - starti + 1, **(obj_args))
          for j in range(stopi - starti + 1):
            objs[j] += lj_repulsion(data[starti + j], ljrmins)
            obj_total += lj_repulsion(data[starti + j], ljrmins)
            objs[j] = objs[j].detach()

          obj_total.backward()
          if (torch.min(objs) < best_obj).item():
            best_obj = torch.min(objs).detach()
            best_x = data[starti + torch.argmin(objs).item()].pos.detach(
              ).flatten()
          del predictions, objs, obj_total
        predicted = True
      except torch.cuda.OutOfMemoryError:
        split -= 1
        assert split >= 0, "Out of memory with only one structure"
    
    if i != niters - 1:
      optimizer.step()

  to_return = best_x.detach().cpu().numpy() 
  del best_obj, best_x, target, data
  return to_return

def basinhop(ensemble, dataset, starts = 128, iters_per_start = 100, 
  lr = 0.01, obj_func = ucb_obj, obj_args = {'λ': 1.0}):
  device = ensemble.device
  ljrmins = torch.tensor(lj_rmins, device = device)

  starting_structures = [dataset.structures[j].copy(
    ) if j < dataset.N else dataset.random_perturbation(
    ) for j in range(starts)]

  new_x = run_adam(ensemble, dataset.target, starting_structures, 
    dataset.config, ljrmins, niters = iters_per_start, lr = lr, 
    mask = dataset.simfunc.mask, obj_func = obj_func, obj_args = obj_args, 
    device = device)
  
  new_structure = starting_structures[0].copy()
  for i in range(len(new_structure)):
    new_structure[i].coords = new_x[(3 * i):(3 * (i + 1))]
  return new_structure
