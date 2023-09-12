import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
import torch
from torch_geometric.data import Data
from matdeeplearn.preprocessor.helpers import (
    generate_edge_features,
    generate_node_features,
    calculate_edges_master,
)

def prepare_data(
    structure, 
    num_offsets: int = 2,
    device: str = "cpu", 
    r: float = 8.0,
    n_neighbors: int = 250,
    edge_steps: int = 50,
    edge_calc_method: str = "mdl",
    all_neighbors: bool = False,
):
    # based on https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/preprocessor/processor.py
    data = Data()
    adaptor = AseAtomsAdaptor()
    n_atoms = len(structure)
    ase_crystal = adaptor.get_atoms(structure)
    data.pos = torch.tensor(ase_crystal.get_positions().tolist(), 
                device = device, dtype = torch.float)
    data.cell = torch.tensor([ase_crystal.get_cell().tolist()], 
                device = device, dtype = torch.float)
    data.z = torch.tensor(ase_crystal.get_atomic_numbers().tolist(), 
                device = device, dtype = torch.long)
    data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])         
    
    edge_gen_out = calculate_edges_master(
        edge_calc_method,
        all_neighbors,
        r,
        n_neighbors,
        num_offsets,
        ["_"],
        data.cell,
        data.pos,
        data.z,
    ) 
                                          
    data.edge_index = edge_gen_out["edge_index"]
    data.edge_vec = edge_gen_out["edge_vec"]
    data.edge_weight = edge_gen_out["edge_weights"]
    data.cell_offsets = edge_gen_out["cell_offsets"]
    data.neighbors = edge_gen_out["neighbors"]            

    if(data.edge_vec.dim() > 2):
        data.edge_vec = data.edge_vec[data.edge_index[0], data.edge_index[1]] 
    
    data.edge_descriptor = {}
    data.edge_descriptor["distance"] = data.edge_weight
    data.distances = data.edge_weight

    generate_node_features(data, n_neighbors, device=device)
    generate_edge_features(data, edge_steps, r, device=device)

    delattr(data, "edge_descriptor")
    data.x = data.x.float()

    return data
