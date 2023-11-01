import torch
import math
from scipy.optimize import basinhopping
import numpy as np
from activestructopt.gnn.dataloader import prepare_data
import periodictable
from pathlib import Path

lj_rmins = np.genfromtxt(str(Path(__file__).parent / "../lj_rmins.csv"), delimiter=",")
el_symbols = [periodictable.elements[i].symbol for i in range(95)]

def get_z(site):
  return np.argmax(el_symbols == site.species.elements[0].symbol)

def get_device(ensemble):
  device = next(iter(ensemble.ensemble[0].trainer.model.state_dict().values(
    ))).get_device()
  device = 'cpu' if device == -1 else 'cuda:' + str(device)
  return device

def generate_data(structure, device, ensemble):
  data = prepare_data(
    structure, ensemble.config['dataset']).to(device)
  data.pos.requires_grad_()
  return data

def lj_repulsion(data, ljrmins, scale = 400):
  rmins = ljrmins[(data.z[data.edge_index[0]] - 1), (data.z[data.edge_index[1]] - 1)]
  return (torch.mean(torch.where(data.edge_weight < rmins, torch.pow(rmins / data.edge_weight, 12), 1)) - 1) / scale

# https://en.wikipedia.org/wiki/Q-function
def Q(x):
  return torch.erfc(x / math.sqrt(2)) / 2

# https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution#Cumulative_distribution_function
def ncx2_cdf(x, λ):
  return 1 - Q(torch.sqrt(x) - torch.sqrt(λ)) - Q(torch.sqrt(x) + torch.sqrt(λ))

# Get PDF as local derivative of CDF
def ncx2_pdf(x, λ, dx = 0.0001):
  return (ncx2_cdf(x + dx, λ) - ncx2_cdf(x - dx, λ)) / (2 * dx)

# Based on https://github.com/JuliaStats/Distributions.jl/blob/master/src/quantilealgs.jl#L50
# if starting at mode, Newton is convergent for any unimodal continuous distribution, see:
#   Göknur Giner, Gordon K. Smyth (2014)
#   A Monotonically Convergent Newton Iteration for the Quantiles of any Unimodal
#   Distribution, with Application to the Inverse Gaussian Distribution
#   http://www.statsci.org/smyth/pubs/qinvgaussPreprint.pdf
def quantile_newton(λ, std, p, eps = 1e-5):
  x = λ + eps
  for i in range(10):
    x = x + (p - ncx2_cdf(x, λ + eps)) / ncx2_pdf(x, λ + eps, dx = 1e-6)
  λ.retain_grad()
  return torch.mean(x * (std ** 2))

def ucb_loss(ensemble, data, target, p = 0.26, device = 'cpu'):
  prediction = ensemble.ensemble[0].trainer.model._forward(data)
  target = torch.tensor(target, device = device)
  for i in range(1, ensemble.k):
    prediction = torch.cat((prediction,
               ensemble.ensemble[i].trainer.model._forward(data)), dim = 0)
  mean = torch.mean(prediction, dim = 0)
  std = ensemble.scalar * torch.std(prediction, dim = 0)
  λs = (mean - target) ** 2 / (std ** 2)
  ucb = quantile_newton(λs, std, p)
  data.pos.retain_grad()
  ucb.backward(retain_graph=True)
  df = data.pos.grad.detach().cpu().numpy().flatten()
  f = ucb.detach().cpu().item()
  return f, df

def old_ucb_loss(ensemble, data, target, λ = 1.0, device = 'cpu'):
  ljrmins = torch.tensor(lj_rmins, device = device)
  prediction = ensemble.ensemble[0].trainer.model._forward(data)
  target = torch.tensor(target, device = device)
  for i in range(1, ensemble.k):
    prediction = torch.cat((prediction,
               ensemble.ensemble[i].trainer.model._forward(data)), dim = 0)
  mean = torch.mean(prediction, dim = 0)
  std = ensemble.scalar * torch.std(prediction, dim = 0)

  yhat = torch.mean((std ** 2) + ((target - mean) ** 2))
  s = torch.sqrt(2 * torch.sum((std ** 4) + 2 * (std ** 2) * (
    (target - mean) ** 2))) / (len(target))
  ucb = yhat - λ * s + lj_repulsion(data, ljrmins)
  data.pos.retain_grad()
  ucb.backward(retain_graph=True)
  df = data.pos.grad.detach().cpu().numpy().flatten()
  f = ucb.detach().cpu().item()
  return f, df

def old_mse_loss(ensemble, data, target, λ = 1.0, device = 'cpu'):
  prediction = ensemble.ensemble[0].trainer.model._forward(data)
  target = torch.tensor(target, device = device)
  for i in range(1, ensemble.k):
    prediction = torch.cat((prediction,
               ensemble.ensemble[i].trainer.model._forward(data)), dim = 0)
  mean = torch.mean(prediction, dim = 0)
  std = ensemble.scalar * torch.std(prediction, dim = 0)

  ucb = torch.mean((std ** 2) + ((target - mean) ** 2))
  data.pos.retain_grad()
  ucb.backward(retain_graph=True)
  df = data.pos.grad.detach().cpu().numpy().flatten()
  f = ucb.detach().cpu().item()
  return f, df

def mse_loss(ensemble, data, target, p = 0.26, device = 'cpu'):
  prediction = ensemble.ensemble[0].trainer.model._forward(data)
  target = torch.tensor(target, device = device)
  for i in range(1, ensemble.k):
    prediction = torch.cat((prediction,
               ensemble.ensemble[i].trainer.model._forward(data)), dim = 0)
  mean = torch.mean(prediction, dim = 0)
  mse = torch.mean((mean - target) ** 2)
  data.pos.retain_grad()
  mse.backward(retain_graph=True)
  df = data.pos.grad.detach().cpu().numpy().flatten()
  f = mse.detach().cpu().item()
  return f, df

def basinhop(ensemble, starting_structure, target, 
    starts = 100, iters_per_start = 100, method = "BFGS", 
    loss_fn = ucb_loss):
  x0 = starting_structure.cart_coords.flatten()
  device = get_device(ensemble)
  def func(x):
    new_structure = starting_structure.copy()
    for i in range(len(new_structure)):
        new_structure[i].coords = x[(3 * i):(3 * (i + 1))]
    f, df = loss_fn(ensemble, 
        generate_data(new_structure, device, ensemble), target, device = device)
    return f, df.tolist()
  def constraint_fun(x):
    new_structure = starting_structure.copy()
    for i in range(len(new_structure)):
        new_structure[i].coords = x[(3 * i):(3 * (i + 1))]
    for i in range(len(new_structure)):
      for j in range(i + 1, len(new_structure)):
        if new_structure.sites[i].distance(
            new_structure.sites[j]) < lj_rmins[get_z(
            new_structure.sites[i])][get_z(new_structure.sites[j])]:
          return False
    return True
  # Based on https://github.com/scipy/scipy/blob/v1.11.3/scipy/optimize/_basinhopping.py#L259C1-L287C17
  class RandomDisplacementWithRejection:
    def __init__(self, stepsize=0.5, random_gen=None):
      self.stepsize = stepsize
    
    def __call__(self, x):
      while True:
        y = x + (2 * np.random.rand(len(x)) - 1)
        if constraint_fun(y):
          return y
  ret = basinhopping(func, x0, minimizer_kwargs = {"method": method, 
    "jac": True, "options": {"maxiter": iters_per_start}}, niter = starts, 
    take_step = RandomDisplacementWithRejection())
  new_structure = starting_structure.copy()
  for i in range(len(new_structure)):
    new_structure[i].coords = ret.x[(3 * i):(3 * (i + 1))]
  return new_structure
