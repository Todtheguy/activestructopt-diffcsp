import torch
import numpy as np
from activestructopt.gnn.dataloader import prepare_data
from activestructopt.optimization.shared.constraints import lj_rmins, lj_repulsion, lj_reject

def run_adam(ensemble, target, starting_structure, config, ljrmins,
                    niters = 100, λ = 1.0, lr = 0.01, device = 'cpu'):
  x0 = torch.tensor(starting_structure.lattice.get_cartesian_coords(
        starting_structure.frac_coords), device = device, dtype = torch.float)
  ucbs = torch.zeros(niters, device = device)
  xs = torch.zeros((niters, 3 * x0.size()[0]), device = device)
  target = torch.tensor(target, device = device)
  data = prepare_data(starting_structure, config, pos_grad = True).to(device)
  data.pos = x0
  optimizer = torch.optim.Adam([data.pos], lr=lr)
  for i in range(niters):
    optimizer.zero_grad(set_to_none=True)
    data.pos.requires_grad_()
    mean, std = ensemble.predict(data, prepared = True)
    yhat = torch.mean((std ** 2) + ((target - mean) ** 2))
    s = torch.sqrt(2 * torch.sum((std ** 4) + 2 * (std ** 2) * (
      (target - mean) ** 2))) / (len(target))
    ucb = yhat - λ * s + lj_repulsion(data, ljrmins)
    if i != niters - 1:
      ucb.backward()
      optimizer.step()
    xs[i] = data.pos.detach().flatten()
    ucbs[i] = ucb.detach().item()
    yhat, s, mean, std, ucb = yhat.detach(), s.detach(
      ), mean.detach(), std.detach(), ucb.detach()
    del yhat, s, mean, std, ucb
    
  to_return = ucbs.detach().cpu().numpy(), xs.detach().cpu().numpy()
  del ucbs, xs, target, data
  to_return = to_return[0][np.argmin(to_return[0])], to_return[1][np.argmin(to_return[0])]
  return to_return

def basinhop(ensemble, starting_structures, target, config,
                  nhops = 10, niters = 100, λ = 1.0, lr = 0.01, 
                  step_size = 0.1, rmcσ = 0.0025):
  device = ensemble.device
  ucbs = np.zeros((nhops))
  xs = np.zeros((nhops, 3 * len(starting_structures[0])))
  ljrmins = torch.tensor(lj_rmins, device = device)

  for i in range(nhops):
    new_ucb, new_x = run_adam(ensemble, target, starting_structures[i], 
      config, ljrmins, niters = niters, λ = λ, lr = lr, device = device)
    
    ucbs[i] = new_ucb
    xs[i] = new_x
  new_structure = starting_structures[0].copy()
  for i in range(len(new_structure)):
    new_structure[i].coords = xs[np.argmin(ucbs)][(3 * i):(3 * (i + 1))]
  return new_structure
