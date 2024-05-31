from activestructopt.common.dataloader import prepare_data, reprocess_data
from activestructopt.common.constraints import lj_rmins, lj_repulsion
from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.common.registry import registry
import torch
import numpy as np

@registry.register_optimizer("Adam")
class Adam(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(model: BaseModel, dataset: BaseDataset, objective: BaseObjective,
    starts = 128, iters_per_start = 100, lr = 0.01, **kwargs):
    
    starting_structures = [dataset.structures[j].copy(
      ) if j < dataset.N else dataset.sample(
      ) for j in range(starts)]
    
    device = model.device
    nstarts = len(starting_structures)
    natoms = len(starting_structures[0])
    ljrmins = torch.tensor(lj_rmins, device = device)
    best_obj = torch.tensor([float('inf')], device = device)
    best_x = torch.zeros(3 * natoms, device = device)
    target = torch.tensor(target, device = device)
    
    data = [prepare_data(s, dataset.config, pos_grad = True, device = device, 
      preprocess = False) for s in starting_structures]
    for i in range(nstarts): # process node features
      reprocess_data(data[i], dataset.config, device, edges = False)
                        
    optimizer = torch.optim.Adam([d.pos for d in data], lr=lr)
    
    split = int(np.ceil(np.log2(nstarts)))
    orig_split = split

    for i in range(iters_per_start):
      predicted = False
      while not predicted:
        try:
          optimizer.zero_grad(set_to_none=True)
          for j in range(nstarts):
            data[j].pos.requires_grad_()
            reprocess_data(data[j], dataset.config, device, nodes = False)

          for k in range(2 ** (orig_split - split)):
            starti = k * (2 ** split)
            stopi = min((k + 1) * (2 ** split) - 1, nstarts - 1)
            predictions = model.predict(data[starti:(stopi+1)], 
              prepared = True, mask = dataset.simfunc.mask)

            objs, obj_total = objective.get(predictions, target, 
              device = device, N = stopi - starti + 1)
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
      
      if i != iters_per_start - 1:
        optimizer.step()

    new_x = best_x.detach().cpu().numpy() 
    del best_obj, best_x, target, data
    new_structure = starting_structures[0].copy()
    for i in range(len(new_structure)):
      new_structure[i].coords = new_x[(3 * i):(3 * (i + 1))]
    return new_structure
