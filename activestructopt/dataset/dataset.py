from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import json

def write_data_splits(initial_structure, folder, optfunc, args, 
                      perturbr = 0.05, N = 100, split = 0.85, k = 5):
  structures = [initial_structure.copy() for i in range(N)]
  for i in range(N):
    structures[i].perturb(perturbr)

  structure_indices = np.random.permutation(np.arange(N))
  trainval_indices = structure_indices[:int(np.floor(split * N))]
  kfolds = np.array_split(trainval_indices, k)
  test_indices = structure_indices[int(np.floor(split * N)):]

  splits_to_make = [('test_data.json', test_indices)]
  for i in range(k):
      splits_to_make.append(('train_k' + str(i) + '.json', np.concatenate([kfolds[i] for i in range(5) if i != k])))
      splits_to_make.append(('val_k' + str(i) + '.json', kfolds[i]))

  for j in range(len(splits_to_make)):
      data_list=[]
      adaptor = AseAtomsAdaptor()
      for i in splits_to_make[j][1]:
          ase_crystal = adaptor.get_atoms(structures[i])
          data_list.append({
              'structure_id': str(i),
              'positions': ase_crystal.get_positions().tolist(),
              'cell': ase_crystal.get_cell().tolist(),
              'atomic_numbers': ase_crystal.get_atomic_numbers().tolist(),
              'y': optfunc(structures[i], **(args)).tolist(),
          })

      with open(folder + '/' + splits_to_make[j][0], 'w') as f:
          json.dump(data_list, f)
  
  return structures, kfolds, test_indices
