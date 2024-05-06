from pymatgen.io import feff
from pymatgen.io.feff.sets import MPEXAFSSet
from pymatgen.io.feff.outputs import Xmu
from pymatgen.core.structure import Structure
import numpy as np
import os

def get_EXAFS(structure, feff_location = "", folder = "", 
	absorber = 'Co', edge = 'K', radius = 10.0, kmax = 12.0, ε = 0.001):
	absorbers = [structure.sites[i] for i in np.argwhere(
		[x.symbol == absorber for x in structure.species]).flatten()]

	assert len(absorbers) > 0

	# guarantees at least two atoms of the absorber,
	# which is necessary because two different ipots are created
	structure.make_supercell(2)

	# get all indices of the absorber
	absorber_indices = -np.ones(len(absorbers), dtype = np.int64)
	for i in range(len(absorbers)):
		for j in range(len(structure.sites)):
			if (abs(absorbers[i].x - structure.sites[j].x) < ε and
					abs(absorbers[i].y - structure.sites[j].y) < ε and
					abs(absorbers[i].z - structure.sites[j].z) < ε):
				absorber_indices[i] = j

	chi_ks = []

	subfolders = [int(x) for x in os.listdir(folder)]
	new_folder = os.path.join(folder, str(np.max(
		subfolders) + 1 if len(subfolders) > 0 else 0))
	os.mkdir(new_folder)
	
	for absorb_ind in absorber_indices:
		new_abs_folder = os.path.join(new_folder, str(absorb_ind))
		os.mkdir(new_abs_folder)

		params = MPEXAFSSet(
			int(absorb_ind),
			structure,
			edge = edge,
			radius = radius,
			user_tag_settings = {'EXAFS': kmax})

		atoms_loc = os.path.join(new_abs_folder, 'ATOMS')
		pot_loc = os.path.join(new_abs_folder, 'POTENTIALS')
		params_loc = os.path.join(new_abs_folder, 'PARAMETERS')

		params.atoms.write_file(atoms_loc)
		params.potential.write_file(pot_loc)
		feff.inputs.Tags(params.tags).write_file(params_loc)
		# https://www.geeksforgeeks.org/python-program-to-merge-two-files-into-a-third-file/
		atoms = pot = tags = ""
		with open(atoms_loc) as fp:
			atoms = fp.read()
		with open(pot_loc) as fp:
			pot = fp.read()
		with open(params_loc) as fp:
			tags = fp.read()
		with open (os.path.join(new_abs_folder, 'feff.inp'), 'w') as fp:
			fp.write(tags + '\n' + pot + '\n' + atoms)
		os.remove('ATOMS')
		os.remove('POTENTIALS')
		os.remove('PARAMETERS')

		# run feff.inp
		os.system(f"cd {new_abs_folder} && {feff_location} feff.inp")

		f = open(os.path.join(new_abs_folder, "xmu.dat"), "r")
		start = 0
		i = 0
		while start == 0:
			i += 1
			if f.readline().startswith("#  omega"):
				start = i
		f.close()

		xmu = Xmu(params.header, feff.inputs.Tags(params.tags), 
			int(absorb_ind), np.genfromtxt(os.path.join(new_abs_folder, "xmu.dat"), 
			skip_header = start))
		
		chi_ks.append(xmu.chi)
	
	return np.mean(np.array(chi_ks), axis = 0)

	