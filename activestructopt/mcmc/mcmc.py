import numpy as np
from scipy.stats import norm

def mcmc_step(structure, tol):
    for i in range(len(structure)):
        structure.sites[i].a = (np.random.uniform(0., tol) ) % 1
        structure.sites[i].b = (np.random.uniform(0., tol) ) % 1
        structure.sites[i].c = (np.random.uniform(0., tol) ) % 1
    return structure

def loglikelihood(exp, th, σ):
    to_return = 0
    assert len(exp) == len(th)
    for i in range(len(exp)):
        to_return += norm.logpdf(exp[i] - th[i], 0, σ)
    return to_return


def mcmc(optfunc, args, exp, structure, N, tol = 0.1):
    # Uniform prior distribution for structure
    for i in range(len(structure)):
        structure.sites[i].a = np.random.uniform(0.,1.)
        structure.sites[i].b = np.random.uniform(0.,1.)
        structure.sites[i].c = np.random.uniform(0.,1.)

    # Uniform prior distribution for noise (TODO: Change this)
    σ = np.random.uniform(0.,1.)

    structures = [structure.copy()]
    loglikelihoods = [loglikelihood(exp, optfunc(structure, **(args)), σ)]
    accepts = [True]
    last_accept = 0

    for i in range(1, N):
        structure = mcmc_step(structure, tol)
        structures.append(structure.copy())
        p = loglikelihood(exp, optfunc(structure, **(args)), σ)
        loglikelihoods.append(p)
        accept = np.log(np.random.uniform(0.,1.)) < (
            p - loglikelihoods[last_accept])
        accepts.append(accept)
        last_accept = i if accept else last_accept
        structure = structure if accept else structures[last_accept]

    return structures, loglikelihoods, accepts