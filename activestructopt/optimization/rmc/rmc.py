import numpy as np
import copy
import math
from scipy.optimize import minimize

def step(structure, latticeprob, σr, σl, σθ):
    new_struct = copy.deepcopy(structure)
    if np.random.rand() < latticeprob:
        lattice_step(new_struct, σl, σθ)
    else:
        positions_step(new_struct, σr)
    return new_struct

def lattice_step(structure, σl, σθ):
    structure.lattice = structure.lattice.from_parameters(
        np.maximum(0.0, structure.lattice.a + σl * np.random.randn()),
        np.maximum(0.0, structure.lattice.b + σl * np.random.randn()), 
        np.maximum(0.0, structure.lattice.c + σl * np.random.randn()), 
        structure.lattice.alpha + σθ * np.random.randn(), 
        structure.lattice.beta + σθ * np.random.randn(), 
        structure.lattice.gamma + σθ * np.random.randn()
    )

def positions_step(structure, σr):
    atom_i = np.random.choice(range(len(structure)))
    structure.sites[atom_i].a = (structure.sites[atom_i].a + 
        σr * np.random.rand() / structure.lattice.a) % 1
    structure.sites[atom_i].b = (structure.sites[atom_i].b + 
        σr * np.random.rand() / structure.lattice.b) % 1
    structure.sites[atom_i].c = (structure.sites[atom_i].c + 
        σr * np.random.rand() / structure.lattice.c) % 1

def mse(exp, th):
    return np.mean((exp - th) ** 2)

def 𝛘2(exp, th, σ):
    return np.mean((exp - th) ** 2) / (σ ** 2)

def reject(structure):
    dists = structure.distance_matrix.flatten()
    return np.min(dists[dists > 0]) < 1

def rmc(optfunc, args, exp, σ, structure, N, latticeprob = 0.1, σr = 0.5, σl = 0.1, σθ = 1.0):
    structures = []
    accepts = []
    old_structure = structure
    old_mse = mse(exp, optfunc(old_structure, **(args)))
    mses = [old_mse]
    Δmses = [-1.]

    for i in range(N):
        new_structure = step(old_structure, latticeprob, σr, σl, σθ)
        res = optfunc(new_structure, **(args))
        new_mse = mse(exp, res)
        Δmse = new_mse - old_mse
        accept = (Δmse <= 0 or np.random.rand() < np.exp(-Δmse/(2 * σ ** 2))) and not reject(new_structure)
        structures.append(new_structure)
        mses.append(new_mse)
        Δmses.append(Δmse)
        accepts.append(accept)
        if accept:
            old_structure = copy.deepcopy(new_structure)
            old_mse = new_mse
        # update σ to achieve 50% acceptance when possible
        if i % 10 == 0:
            recent_Δmses = np.array(Δmses[-10:])
            increases = recent_Δmses[recent_Δmses > 0]
            if len(increases) <= 5:
                continue
            expectation_target = 0.5 - ((10 - len(increases)) / 10)
            f = lambda x: np.abs(expectation_target - np.sum(np.exp(-increases/(2 * x[0] ** 2))) / 10)
            σ = minimize(f, [σ]).x[0]

    return structures, mses, accepts

def 𝛘2_ucb(exp, th, thσ, σ, λ):
    # noncentral chi squared distributions for each dimension
    yhats = (th - exp) ** 2 / (thσ ** 2) + np.ones(len(exp))
    ss = 2 * (np.ones(len(exp)) + 2 * (th - exp) ** 2 / (thσ ** 2))
    return np.mean(yhats - λ * ss) / (σ ** 2)

def rmc_ucb(optfunc, args, exp, σ, structure, N, σr = 0.1, λ = 1.0):
    structures = []
    𝛘2s = []
    old_structure = structure
    res, resσ = optfunc(old_structure, **(args))
    old_𝛘2 = 𝛘2_ucb(exp, res, resσ, σ, λ)

    for _ in range(N):
        new_structure = step(old_structure, 0.0, σr, 0.0, 0.0)
        res, resσ = optfunc(new_structure, **(args))
        new_𝛘2 = 𝛘2_ucb(exp, res, resσ, σ, λ)
        Δχ2 = new_𝛘2 - old_𝛘2
        accept = np.random.rand() < np.exp(-Δχ2/2) and not reject(new_structure)
        structures.append(new_structure)
        𝛘2s.append(new_𝛘2)
        if accept:
            old_structure = copy.deepcopy(new_structure)
            old_𝛘2 = new_𝛘2

    return structures[np.argmin(𝛘2s)]

def rmc_exploit(optfunc, args, exp, σ, structure, N, σr = 0.1, λ = 1.0):
    structures = []
    𝛘2s = []
    old_structure = structure
    res, resσ = optfunc(old_structure, **(args))
    old_𝛘2 = 𝛘2_ucb(exp, res, resσ, σ, λ)

    for _ in range(N):
        new_structure = step(old_structure, 0.0, σr, 0.0, 0.0)
        res, resσ = optfunc(new_structure, **(args))
        new_𝛘2 = 𝛘2(exp, res, σ)
        Δχ2 = new_𝛘2 - old_𝛘2
        accept = np.random.rand() < np.exp(-Δχ2/2) and not reject(new_structure)
        structures.append(new_structure)
        𝛘2s.append(new_𝛘2)
        if accept:
            old_structure = copy.deepcopy(new_structure)
            old_𝛘2 = new_𝛘2

    return structures[np.argmin(𝛘2s)]
