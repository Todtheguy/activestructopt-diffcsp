from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import json

def write_data_splits(initial_structure, folder, optfunc, args, 
                      perturbr = 0.05, N = 100, splits = (0.8, 0.05, 0.15)):
  structures = [initial_structure.copy() for i in range(N)]
  for i in range(N):
    structures[i].perturb(perturbr)

  structure_indices = np.random.permutation(np.arange(N))
  train_indices = structure_indices[:int(np.floor(splits[0] * N))]
  val_indices = structure_indices[int(np.floor(splits[0] * N)):int(np.floor((splits[0] + splits[1]) * N))]
  test_indices = structure_indices[int(np.floor((splits[0] + splits[1]) * N)):]

  data_list=[]
  adaptor = AseAtomsAdaptor()

  for i in train_indices:
      ase_crystal = adaptor.get_atoms(structures[i])
      data_list.append({
          'structure_id': str(i),
          'positions': ase_crystal.get_positions().tolist(),
          'cell': ase_crystal.get_cell().tolist(),
          'atomic_numbers': ase_crystal.get_atomic_numbers().tolist(),
          'y': optfunc(structures[i], **(args)).tolist(),
      })

  with open(folder + '/train_data.json', 'w') as f:
      json.dump(data_list, f)

  for i in val_indices:
      ase_crystal = adaptor.get_atoms(structures[i])
      data_list.append({
          'structure_id': str(i),
          'positions': ase_crystal.get_positions().tolist(),
          'cell': ase_crystal.get_cell().tolist(),
          'atomic_numbers': ase_crystal.get_atomic_numbers().tolist(),
          'y': optfunc(structures[i], **(args)).tolist(),
      })

  with open(folder + '/val_data.json', 'w') as f:
      json.dump(data_list , f)

  for i in test_indices:
      ase_crystal = adaptor.get_atoms(structures[i])
      data_list.append({
          'structure_id': str(i),
          'positions': ase_crystal.get_positions().tolist(),
          'cell': ase_crystal.get_cell().tolist(),
          'atomic_numbers': ase_crystal.get_atomic_numbers().tolist(),
          'y': optfunc(structures[i], **(args)).tolist(),
      })

  with open(folder + '/test_data.json', 'w') as f:
      json.dump(data_list, f)
  
  return structures, train_indices, val_indices, test_indices
