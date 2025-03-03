import sys
sys.path.append("/home/hice1/adaftardar3/ActiveStructOpt-Forked/ActiveStructOpt")
from activestructopt.simulation.rdf import RDF
from activestructopt.active.config import torchmd_diffusion_config
from activestructopt.active.active import ActiveLearning
import copy
import numpy as np
import os
from pymatgen.io.cif import CifParser

def match_files(directory, number):
    # Convert the number to a string
    number_str = f"{number}_"
    # List all files in the directory
    all_files = os.listdir(directory)
    # Filter files that start with the given number and an underscore
    matching_files = [file for file in all_files if file.startswith(number_str)]
    return matching_files[0]

def main():
    config = torchmd_diffusion_config
    pristine_structure = CifParser("/home/hice1/adaftardar3/ActiveStructOpt-Forked/ActiveStructOpt/activestructopt/testing/datasets/ht/start/" + str(sys.argv[1]) + ".cif").get_structures(primitive = False)[0]
    target_structure = CifParser("/home/hice1/adaftardar3/ActiveStructOpt-Forked/ActiveStructOpt/activestructopt/testing/datasets/ht/target/" + str(sys.argv[1]) + ".cif").get_structures(primitive = False)[-1]

    rdf_func = RDF(pristine_structure, σ = 0.1, max_r = 12.)

    target_promise = copy.deepcopy(rdf_func)
    target_promise.get(target_structure)
    target_spec = target_promise.resolve()
    target_spec = np.mean(target_spec[np.array(rdf_func.mask)], axis = 0)
    progress_file = None
    if len(sys.argv) == 3 and str(sys.argv[2]) == "True":
        directory  = '/home/hice1/adaftardar3/ActiveStructOpt-Forked/ActiveStructOpt/activestructopt/testing/progress/'
        progress_file = directory + match_files(directory, sys.argv[1])
    al = ActiveLearning(rdf_func, target_spec, config, pristine_structure, 
        index = sys.argv[1], target_structure = target_structure, progress_file=progress_file)
    al.optimize(save_progress_dir = '/home/hice1/adaftardar3/ActiveStructOpt-Forked/ActiveStructOpt/activestructopt/testing/progress/')
    al.save("/home/hice1/adaftardar3/ActiveStructOpt-Forked/ActiveStructOpt/activestructopt/testing/res/" + str(sys.argv[1]) + ".pkl", 
        additional_data = {'target_structure': target_structure})

if __name__ == "__main__":
    main()
