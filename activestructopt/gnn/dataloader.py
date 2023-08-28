import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as DataLoader

class PrepareData(Dataset):
    def __init__(
        self,
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
        self.data = Data()

        n_atoms = len(structure)
        ase_crystal = adaptor.get_atoms(structure)
        self.data.pos = torch.tensor(ase_crystal.get_positions().tolist(), 
                    device = device, dtype = torch.float)
        self.data.cell = torch.tensor([ase_crystal.get_cell().tolist()], 
                    device = device, dtype = torch.float)
        self.data.z = torch.tensor(ase_crystal.get_atomic_numbers().tolist(), 
                    device = device, dtype = torch.long)
        self.data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])         
        
        edge_gen_out = calculate_edges_master(
            edge_calc_method,
            all_neighbors,
            r,
            n_neighbors,
            num_offsets,
            ["_"],
            self.data.cell,
            self.data.pos,
            self.data.z,
        )
        edge_indices = edge_gen_out["edge_index"]
        edge_weights = edge_gen_out["edge_weights"]
        cell_offsets = edge_gen_out["cell_offsets"]
        edge_vec = edge_gen_out["edge_vec"]
        neighbors = edge_gen_out["neighbors"]
        if(edge_vec.dim() > 2):
            edge_vec = edge_vec[edge_indices[0], edge_indices[1]]  
                                              
        self.data.edge_index, self.data.edge_weight = edge_indices, edge_weights
        self.data.edge_vec = edge_vec
        self.data.cell_offsets = cell_offsets
        self.data.neighbors = neighbors            

        self.data.edge_descriptor = {}
        self.data.edge_descriptor["distance"] = edge_weights
        self.data.distances = edge_weights

        generate_node_features(self.data, n_neighbors, device=device)
        generate_edge_features(self.data, edge_steps, r, device=device)

        Compose([])(self.data)
        delattr(self.data, "edge_descriptor")
        self.data.x = self.data.x.float()

    def __getitem__(self, index):
        return self.data

    def __len__(self):
        return 1

class DataWrapper(object):
    # based on https://github.com/AkshayIyer/CS8803_MLC_Project/blob/shuyi/dataset_utils/datasets.py
    def __init__(self, batch_size, num_workers, **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_dataloader(self,
        structure,
        device: str = "cpu",
        r: float = 8.0,
        n_neighbors: int = 250,
        edge_steps: int = 50,):

        return DataLoader(
            PrepareData(structure, device = device, r = r, 
                        n_neighbors = n_neighbors, edge_steps = edge_steps),
            batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=False,
        )
