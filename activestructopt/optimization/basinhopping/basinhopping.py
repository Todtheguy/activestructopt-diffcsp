import torch
import numpy as np
from activestructopt.gnn.dataloader import prepare_data, reprocess_data, reprocess_data_for_opt_check
from activestructopt.optimization.shared.constraints import lj_rmins, lj_repulsion
from matdeeplearn.preprocessor.helpers import calculate_edges_master

def run_adam(ensemble, target, starting_structures, config, ljrmins,
                    niters = 100, λ = 1.0, lr = 0.01, device = 'cpu'):
  torch.autograd.set_detect_anomaly(True)
  nstarts = len(starting_structures)
  natoms = len(starting_structures[0])
  best_ucb = torch.tensor([float('inf')], device = device)
  best_x = torch.zeros(3 * natoms, device = device)
  target = torch.tensor(target, device = device)
  data = [prepare_data(s, config, pos_grad = True, device = device, preprocess = False) for s in starting_structures]
  for i in range(nstarts):
    #data[i].pos = torch.tensor(starting_structures[i].lattice.get_cartesian_coords(
    #    starting_structures[i].frac_coords), device = device, dtype = torch.float)
    reprocess_data_for_opt_check(data[i], config, device, edges = False)
  
  data[0].pos.requires_grad_()                    
  optimizer = torch.optim.Adam([data[0].pos], lr=lr)
  optimizer.zero_grad(set_to_none=True)
  
  r = config['preprocess_params']['cutoff_radius']
  n_neighbors = config['preprocess_params']['n_neighbors']

  if config['preprocess_params']['preprocess_edges']:
    edge_gen_out = calculate_edges_master(
      config['preprocess_params']['edge_calc_method'],
      r,
      n_neighbors,
      config['preprocess_params']['num_offsets'],
      ["_"],
      data[0].cell,
      data[0].pos,
      data[0].z,
      device = device
    ) 

    ucb = torch.sum(edge_gen_out["edge_weights"])
    ucb.backward(retain_graph=True)
    assert False

  large_structure = False

  for i in range(niters):
    print(i)
    optimizer.zero_grad(set_to_none=True)
    for j in range(nstarts):
      data[j].pos.requires_grad_()
      r = config['preprocess_params']['cutoff_radius']
      n_neighbors = config['preprocess_params']['n_neighbors']
  
      if config['preprocess_params']['preprocess_edges']:
        edge_gen_out = calculate_edges_master(
          config['preprocess_params']['edge_calc_method'],
          r,
          n_neighbors,
          config['preprocess_params']['num_offsets'],
          ["_"],
          data[j].cell,
          data[j].pos,
          data[j].z,
          device = device
        ) 

        ucb = torch.sum(edge_gen_out["edge_weights"])
        ucb.backward()
        assert False
                                                
        data[j].edge_index = edge_gen_out["edge_index"]
        data[j].edge_vec = edge_gen_out["edge_vec"]
        data[j].edge_weight = edge_gen_out["edge_weights"]
        data[j].cell_offsets = edge_gen_out["cell_offsets"]
        data[j].neighbors = edge_gen_out["neighbors"]            
      
        if(data[j].edge_vec.dim() > 2):
          data[j].edge_vec = data[j].edge_vec[data[j].edge_index[0], data[j].edge_index[1]]

    if not large_structure:
      try:
        #predictions = ensemble.predict(data, prepared = True)
        #ucbs = torch.zeros(nstarts)
        #ucb_total = torch.tensor([0.0], device = device)
        #for j in range(1):
        #yhat = torch.mean((predictions[1][j] ** 2) + ((target - predictions[0][j]) ** 2))
        #s = torch.sqrt(2 * torch.sum((predictions[1][j] ** 4) + 2 * (predictions[1][j] ** 2) * (
        #  (target - predictions[0][j]) ** 2))) / (len(target))
        #ucb = yhat - λ * s + lj_repulsion(data[j], ljrmins)
        ucb = lj_repulsion(data[j], ljrmins)
        #ucb_total = ucb_total + ucb
        #ucbs[j] = ucb.clone().detach()
        ucb.backward()
        del predictions, ucb, yhat, s
      except torch.cuda.OutOfMemoryError:
        large_structure = True

    if large_structure:
      ucbs = torch.zeros(nstarts)
      for j in range(nstarts):
        predictions = ensemble.predict([data[j]], prepared = True)
        yhat = torch.mean((predictions[1][0] ** 2) + ((target - predictions[0][0]) ** 2))
        s = torch.sqrt(2 * torch.sum((predictions[1][0] ** 4) + 2 * (predictions[1][0] ** 2) * (
          (target - predictions[0][0]) ** 2))) / (len(target))
        ucb = yhat - λ * s + lj_repulsion(data[j], ljrmins)
        ucbs[j] = ucb.detach()
        ucb.backward()
        del predictions, yhat, s, ucb
    
    if i != niters - 1:
      optimizer.step()
    if (torch.min(ucbs) < best_ucb).item():
      best_ucb = torch.min(ucbs).detach()
      best_x = data[torch.argmin(ucbs).item()].pos.detach().flatten()
    # data[j].pos = data[j].pos.detach()
    del ucbs
    
  to_return = best_x.detach().cpu().numpy() 
  del best_ucb, best_x, target, data
  return to_return

def basinhop(ensemble, starting_structures, target, config,
                  nhops = 10, niters = 100, λ = 1.0, lr = 0.01, 
                  step_size = 0.1, rmcσ = 0.0025):
  device = ensemble.device
  ljrmins = torch.tensor(lj_rmins, device = device)

  new_x = run_adam(ensemble, target, starting_structures, config, ljrmins, 
    niters = niters, λ = λ, lr = lr, device = device)
  
  new_structure = starting_structures[0].copy()
  for i in range(len(new_structure)):
    new_structure[i].coords = new_x[(3 * i):(3 * (i + 1))]
  return new_structure
