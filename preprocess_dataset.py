import torch
import torch_geometric as ptg
import matplotlib.pyplot as plt
import numpy as np
import argparse
import networkx as nx
from torch_sparse import coalesce

import visualization as vis
import constants
import utils
from data_loading.california import load_cal
from data_loading.wind import load_wind_speed, load_wind_cap
from data_loading.wiki import wiki_loader

DATASETS = {
    "cal": load_cal,
    "wind_speed": load_wind_speed,
    "wind_cap": load_wind_cap,
    "wiki_squirrel": wiki_loader("squirrel"),
    "wiki_chameleon": wiki_loader("chameleon"),
    "wiki_crocodile": wiki_loader("crocodile")
}

# Always on cpu
parser = argparse.ArgumentParser(description='Pre-process dataset')
parser.add_argument("--dataset", type=str, help="Dataset to pre-process")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--plot", type=int, default=0,
        help="If plots should be made during generation")

parser.add_argument("--graph_alg", type=str, default="delaunay",
        help="Algorithm to use for constructing graph")
parser.add_argument("--n_neighbors", type=int, default=5,
        help="Amount of neighbors to include in k-nn graph generation")

parser.add_argument("--mask_fraction", type=float, default=0.25,
        help="Fraction to mask when using random_mask")
parser.add_argument("--random_mask", type=int, default=1,
        help="Use a random mask rather than cut-out areas")

parser.add_argument("--dist_weight", type=int, default=0,
        help="Compute also eigenvalues for distance weighting")
parser.add_argument("--dist_weight_eps", type=int, default=1e-2,
        help="Epsilon to add to distances to prevent division by zero")

# log-det pre-processing steps
parser.add_argument("--compute_eigvals", type=int, default=1,
        help="If eigenvalues should be computed")
parser.add_argument("--compute_dad_traces", type=int, default=0,
        help="If traces of the DAD-matrix should be estimated")
parser.add_argument("--dad_samples", type=int, default=1000,
        help="Amount of samples to use in DAD trace estimates")
parser.add_argument("--dad_k_max", type=int, default=50,
        help="Maximum k to compute DAD trace for")

args = parser.parse_args()

_ = torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

assert args.dataset, "No dataset selected"
assert args.dataset in DATASETS, "Unknown dataset: {}".format(args.dataset)

# Load dataset
print("Loading dataset ...")
load_return = DATASETS[args.dataset](args)

if type(load_return) == ptg.data.Data:
    # Data loading created full graph
    graph_y = load_return

    if hasattr(graph_y, "mask_limits"):
        mask_limits = graph_y.mask_limits
    else:
        mask_limits = None

    full_ds_name = args.dataset

else:
    # Data loading only set up features, y and positions (i.e. spatial data)
    X_features, y, pos, mask_limits = load_return

    # Turn everything into pytorch tensors
    pos = torch.tensor(pos, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Generate graphs
    print("Generating graph ...")
    point_data = ptg.data.Data(pos=pos)

    if args.graph_alg == "delaunay":
        graph_transforms = ptg.transforms.Compose((
            ptg.transforms.Delaunay(),
            ptg.transforms.FaceToEdge(),
        ))
        graph_y = graph_transforms(point_data)
        full_ds_name = args.dataset + "_delaunay"
    elif args.graph_alg == "knn":
        graph_transforms = ptg.transforms.Compose((
            ptg.transforms.KNNGraph(k=args.n_neighbors, force_undirected=True),
        ))
        graph_y = graph_transforms(point_data)
        full_ds_name = "{}_{}nn".format(args.dataset, args.n_neighbors)
    elif args.graph_alg == "radknn":
        # Add first neighbors within radius, then k-nn until a minimum is reached

        # Compute 95%-quantile of distance to closest neighbor
        nn_transforms = ptg.transforms.Compose((
            ptg.transforms.KNNGraph(k=1, force_undirected=False),
        ))
        nn_graph = nn_transforms(point_data)

        distances = torch.norm(nn_graph.pos[nn_graph.edge_index[0,:]] -\
                nn_graph.pos[nn_graph.edge_index[1,:]], dim=-1).numpy()
        r = np.quantile(distances, q=0.95)

        # Create radius graph
        rad_transforms = ptg.transforms.Compose((
            ptg.transforms.RadiusGraph(r=r, max_num_neighbors=100), # High max
        ))
        rad_edges = rad_transforms(point_data).edge_index

        # Compute k-nn edges
        knn_transforms_ud = ptg.transforms.Compose((
            ptg.transforms.KNNGraph(k=args.n_neighbors, force_undirected=True),
        ))
        knn_edges = knn_transforms_ud(point_data).edge_index

        # Join rad and knn edge-index
        joined_edge_index = torch.cat((rad_edges, knn_edges), axis=1)
        # Remove duplicate edges
        joined_edge_index, _ = coalesce(joined_edge_index, None,
                point_data.num_nodes, point_data.num_nodes)
        graph_y = point_data
        graph_y.edge_index = joined_edge_index

        full_ds_name = "{}_rad{}nn".format(args.dataset, args.n_neighbors)
    else:
        assert False, "Unknown graph algorithm"

    if len(y.shape) == 1:
        # Make sure y tensor has 2 dimensions
        y = y.unsqueeze(1)
    graph_y.x = y

    if not utils.is_none(X_features):
        X_features = torch.tensor(X_features, dtype=torch.float32)
        graph_y.features = X_features

# Check if graph is connected or contains isolated components
nx_graph = ptg.utils.to_networkx(graph_y, to_undirected=True)
n_components = nx.number_connected_components(nx_graph)
contains_iso = ptg.utils.contains_isolated_nodes(graph_y.edge_index, graph_y.num_nodes)
print("Graph connected: {}, n_components: {}".format((n_components == 1), n_components))
print("Contains isolated components (1-degree nodes): {}".format(contains_iso))

# Create Mask
if args.random_mask:
    n_mask = int(args.mask_fraction*graph_y.num_nodes)
    unobs_indexes = torch.randperm(graph_y.num_nodes)[:n_mask]

    unobs_mask = torch.zeros(graph_y.num_nodes).to(bool)
    unobs_mask[unobs_indexes] = True
else:
    assert not utils.is_none(mask_limits), "No mask limits exists for dataset"
    unobs_masks = torch.stack([torch.bitwise_and(
        (graph_y.pos >= limits[0]),
        (graph_y.pos < limits[1])
        ) for limits in mask_limits], dim=0) # Shape (n_masks, n_nodes, 2)

    unobs_mask = torch.any(torch.all(unobs_masks, dim=2), dim=0) # Shape (n_nodes,)

    graph_y.mask_limits = mask_limits

obs_mask = unobs_mask.bitwise_not()

n_masked = torch.sum(unobs_mask)
print("Masked {} / {} nodes".format(n_masked, graph_y.num_nodes))

graph_y.mask = obs_mask

if args.plot:
    vis.plot_graph(graph_y, "y", show=True, title="y")

# Additional computation if weighting by node distances
if args.dist_weight:
    utils.dist_weight_graph(graph_y, args.dist_weight_eps,
            compute_eigvals=bool(args.compute_eigvals))

# Log-determinant pre-processing steps
utils.log_det_preprocess(graph_y, args.dist_weight, args.compute_eigvals,
        args.compute_dad_traces, args.dad_k_max, args.dad_samples)

if args.compute_eigvals and args.plot:
    # Plot histogram of eigenvalues
    plt.hist(graph_y.eigvals.numpy(), bins=100, range=(-1,1))
    plt.title("Histogram of eigenvalues")
    plt.show()

# Save dataset
print("Saving graphs ...")
if args.random_mask:
    full_ds_name += "_random_{}".format(args.mask_fraction)

utils.save_graph_ds({"graph_y": graph_y}, args, full_ds_name)

