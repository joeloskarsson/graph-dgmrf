import argparse
import torch
import numpy as np
import networkx as nx
import torch_geometric as ptg

import utils

parser = argparse.ArgumentParser(
        description='Print graph statistics')
parser.add_argument("--dataset", type=str, default="cal_delaunay_random_0.5",
        help="Dataset to print statistics for")
args = parser.parse_args()

loaded_data = utils.load_dataset(args.dataset)
graph = loaded_data["graph_y"]

n_nodes = graph.num_nodes
n_edges = graph.num_edges / 2 # Undirected
density = (2*n_edges) / (n_nodes*(n_nodes-1))

masked_frac = 1. - (torch.sum(graph.mask)/n_nodes)

nx_g = ptg.utils.to_networkx(graph, to_undirected=True, remove_self_loops=True)
diam = nx.algorithms.distance_measures.diameter(nx_g, usebounds=True)

print("Dataset {}".format(args.dataset))
print("{} nodes".format(n_nodes))
print("{} edges".format(n_edges))
print("Density {:.7}".format(density))
print("Diameter {}".format(diam))
print("{:.3} of nodes masked".format(masked_frac))

