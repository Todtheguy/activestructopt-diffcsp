from activestructopt.simulation.rdf import RDF
from activestructopt.active.config import torchmd_config
from activestructopt.active.active import ActiveLearning
import copy
import numpy as np
import os
import sys
from pymatgen.io.cif import CifParser

def main():
	config = torchmd_config
	pristine_structure = CifParser("~/ActiveStructOpt/activestructopt/testing/datasets/ht/start/" + str(sys.argv[1]) + ".cif"
		).get_structures(primitive = False)[0]
	target_structure = CifParser("~/ActiveStructOpt/activestructopt/testing/datasets/ht/target/" + str(sys.argv[1]) + ".cif"
		).get_structures(primitive = False)[-1]

	rdf_func = RDF(pristine_structure, σ = 0.1, max_r = 12.)

	target_promise = copy.deepcopy(rdf_func)
	target_promise.get(target_structure)
	target_spec = target_promise.resolve()
	target_spec = np.mean(target_spec[np.array(rdf_func.mask)], axis = 0)

	al = ActiveLearning(rdf_func, target_spec, config, pristine_structure, 
		index = sys.argv[1], target_structure = target_structure)
	al.optimize(save_progress_dir = 'progress')
	al.save("~/ActiveStructOpt/activestructopt/testing/res/" + str(sys.argv[1]) + ".pkl", 
		additional_data = {'target_structure': target_structure})

if __name__ == "__main__":
    main()
