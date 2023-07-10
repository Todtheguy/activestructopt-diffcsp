# https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/

import nlopt
from numpy import *
import numpy as np

def run_nlopt(optfunc, args, exp, structure, N):
    natoms = len(structure)

    xstart = []
    for i in range(natoms):
        xstart.append(structure.sites[i].a)
        xstart.append(structure.sites[i].b)
        xstart.append(structure.sites[i].c)

    def modify_structure(x):
        for i in range(natoms):
            structure.sites[i].a = x[3 * i]
            structure.sites[i].b = x[3 * i + 1]
            structure.sites[i].c = x[3 * i + 2]

    def f(x, grad):
        assert not (grad.size > 0)
        modify_structure(x)
        to_return = np.mean((exp - optfunc(structure, **(args))) ** 2)
        return to_return

    opt = nlopt.opt(nlopt.GN_ISRES, 3 * natoms)
    opt.set_min_objective(f)
    opt.set_lower_bounds(np.zeros(3 * natoms))
    opt.set_upper_bounds(np.ones(3 * natoms))
    opt.set_maxeval(N)
    return opt.optimize(xstart)


