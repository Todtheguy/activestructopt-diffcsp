import torch
import numpy as np
from activestructopt.gnn.dataloader import prepare_data
from activestructopt.optimization.shared.constraints import lj_rmins, lj_repulsion, lj_reject

def run_adam(ensemble, target, x0, starting_structure, config, ljrmins,
                    niters = 100, λ = 1.0, lr = 0.01, device = 'cpu'):
  ucbs = torch.zeros(niters, device = device)
  xs = torch.zeros((niters, 3 * x0.size()[0]), device = device)
  target = torch.tensor(target, device = device)
  data = prepare_data(starting_structure, config, pos_grad = True).to(device)
  data.pos = x0
  optimizer = torch.optim.Adam([data.pos], lr=lr)
  for j in range(niters):
    optimizer.zero_grad(set_to_none=True)
    data.pos.requires_grad_()
    prediction = ensemble.ensemble[0].trainer.model._forward(data)
    for i in range(1, ensemble.k):
      prediction = torch.cat((prediction,
                ensemble.ensemble[i].trainer.model._forward(data)), dim = 0)
    mean = torch.mean(prediction, dim = 0)
    std = ensemble.scalar * torch.std(prediction, dim = 0) * np.sqrt((ensemble.k - 1) / ensemble.k)
    yhat = torch.mean((std ** 2) + ((target - mean) ** 2))
    s = torch.sqrt(2 * torch.sum((std ** 4) + 2 * (std ** 2) * (
      (target - mean) ** 2))) / (len(target))
    ucb = yhat - λ * s + lj_repulsion(data, ljrmins)
    if j != niters - 1:
      ucb.backward()
      optimizer.step()
    xs[j] = data.pos.detach().flatten()
    ucbs[j] = ucb.detach().item()
    yhat, s, mean, std, prediction, ucb = yhat.detach(), s.detach(), mean.detach(), std.detach(), prediction.detach(), ucb.detach()
    del yhat, s, mean, std, prediction, ucb
    
  to_return = ucbs.detach().cpu().numpy(), xs.detach().cpu().numpy()
  del ucbs, xs, target, data
  return to_return

def basinhop(ensemble, starting_structure, target, config,
                  nhops = 10, niters = 100, λ = 1.0, lr = 0.01, 
                  step_size = 0.1, rmcσ = 0.0025):
  device = ensemble.device
  ucbs = np.zeros((nhops, niters))
  xs = np.zeros((nhops, niters, 3 * len(starting_structure)))
  ljrmins = torch.tensor(lj_rmins, device = device)

  x0 = torch.tensor(starting_structure.lattice.get_cartesian_coords(
    starting_structure.frac_coords), device = device, dtype = torch.float)

  for j in range(nhops):
    new_ucbs, new_xs = run_adam(ensemble, target, x0, starting_structure, 
      config, ljrmins, niters = niters, λ = λ, lr = lr, device = device)
    
    ucbs[j] = new_ucbs
    xs[j] = new_xs
    if not j + 1 == nhops:
      accepted = xs[j][-1] if j == 0 or np.log(np.random.rand()) < (
        ucbs[j - 1][-1] - ucbs[j][-1]) / (2 * rmcσ ** 2) else accepted
      rejected = True
      while rejected:
        hop = starting_structure.copy()
        for i in range(len(hop)):
          hop[i].coords = accepted[(3 * i):(3 * (i + 1))]
        hop.perturb(step_size)
        rejected = lj_reject(hop)
      x0 = torch.tensor(hop.lattice.get_cartesian_coords(hop.frac_coords), 
        device = device, dtype = torch.float)
  hop, iteration = np.unravel_index(np.argmin(ucbs), ucbs.shape)
  newstructure = starting_structure.copy()
  for i in range(len(newstructure)):
    newstructure[i].coords = xs[hop][iteration][(3 * i):(3 * (i + 1))]
  return newstructure
