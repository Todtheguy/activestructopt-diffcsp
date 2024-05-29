import torch

def ucb_obj(predictions, target, device = 'cpu', N = 1, λ = 1.0):
	ucbs = torch.zeros(N, device = device)
	ucb_total = torch.tensor([0.0], device = device)
	for i in range(N):
		yhat = torch.mean((predictions[1][i] ** 2) + (
			(target - predictions[0][i]) ** 2))
		s = torch.sqrt(2 * torch.sum((predictions[1][i] ** 4) + 2 * (
			predictions[1][i] ** 2) * ((
			target - predictions[0][i]) ** 2))) / (len(target))
		ucb = torch.maximum(yhat - λ * s, torch.tensor(0.))
		ucb_total = ucb_total + ucb
		ucbs[i] = ucb.detach()
		del yhat, s, ucb
	return ucbs, ucb_total

def mse_obj(predictions, target, device = 'cpu', N = 1):
	mses = torch.zeros(N, device = device)
	mse_total = torch.tensor([0.0], device = device)
	for i in range(N):
		mse = torch.mean((target - predictions[0][i]) ** 2)
		mse_total = mse_total + mse
		mses[i] = mse.detach()
		del mse
	return mses, mse_total

def mae_obj(predictions, target, device = 'cpu', N = 1):
	maes = torch.zeros(N, device = device)
	mae_total = torch.tensor([0.0], device = device)
	for i in range(N):
		mae = torch.mean(torch.abs(target - predictions[0][i]))
		mae_total = mae_total + mae
		maes[i] = mae.detach()
		del mae
	return maes, mae_total
