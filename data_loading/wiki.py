import numpy as np
import torch
import os
import torch_geometric as ptg

import constants

DATA_DIR = "wikipedia"
EDGE_FILE = "{}_edges.csv"
TARGET_FILE = "{}_target.csv"

THEMES = ["chameleon", "crocodile", "squirrel"]

wiki_loader = (lambda theme: (lambda args: load_wiki(theme)))

def load_wiki(theme):
    assert theme in THEMES, "Unknown wiki dataset"

    # Load files
    theme_data_dir = DATA_DIR.format(theme)
    theme_edge_file = EDGE_FILE.format(theme)
    theme_target_file = TARGET_FILE.format(theme)

    edge_path = os.path.join(constants.RAW_DATA_DIR, theme_data_dir, theme_edge_file)
    target_path = os.path.join(constants.RAW_DATA_DIR, theme_data_dir, theme_target_file)

    edges_mat = np.genfromtxt(edge_path, delimiter=",", skip_header=1)
    targets_mat = np.genfromtxt(target_path, delimiter=",", skip_header=1)

    edges = edges_mat.T # Flip for ptg
    targets = np.log(targets_mat[:,1]) # Disregard id column, take log of traffic

    num_nodes = len(targets)
    pos = torch.rand(num_nodes, 2)

    # Make torch tensors
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    edge_index = torch.tensor(edges, dtype=torch.long)

    # Make undirected, remove self-loops
    edge_index = ptg.utils.to_undirected(edge_index)
    edge_index = ptg.utils.remove_self_loops(edge_index)[0]

    return ptg.data.Data(x=y, pos=pos, edge_index=edge_index)

