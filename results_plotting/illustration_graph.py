import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch_geometric as ptg
import sys
import wandb
sys.path.insert(0,'..')

import utils
import visualization as vis

OUTPUT_DIR = "illustration_graphs"

SEED = 4547
DS_NAME = "toy_gmrf4547_3_layers_random"
N_LAYERS = 3
ALPHA = 1.5
BETA = -1.

noise_std = 0.01

np.random.seed(SEED)
torch.manual_seed(SEED)

data = utils.load_dataset(DS_NAME, ds_dir="../dataset")
graph_x = data["graph_x"]
x = graph_x.x
N = graph_x.num_nodes

# Make y
y = x + noise_std*torch.randn(N,1)

# Make h:s and z
A = ptg.utils.to_dense_adj(graph_x.edge_index)[0]
d = torch.sum(A, dim=0)
D = torch.diag(d)
G = ALPHA*D + BETA*A # gamma = 0

h1 = G@x
h2 = G@h1
z = G@h2

graph_h1 = utils.new_graph(graph_x, new_x=h1)
graph_h2 = utils.new_graph(graph_x, new_x=h2)
graph_z = utils.new_graph(graph_x, new_x=z)

plots = [
    (x, "x"),
    (y, "y"),
    (h1, "h1"),
    (h2, "h2"),
    (z, "z"),
]

for v, name in plots:
    graph = utils.new_graph(graph_x, new_x=v)
    fig = vis.plot_graph(graph, name, return_plot=True, node_size=1000, title=name)

    save_path = os.path.join(OUTPUT_DIR, "{}.pdf".format(name))
    fig.savefig(save_path)

