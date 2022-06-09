import argparse
import torch
import numpy as np

import utils

MASK_FRACTIONS = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)

parser = argparse.ArgumentParser(
        description='Create additional versions of masked dataset')
parser.add_argument("--dataset", type=str,
        help="Dataset to create additional masks for")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()

assert args.dataset, "Must specify dataset"

# Seed
_ = torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

loaded_data = utils.load_dataset(args.dataset)
graph_y = loaded_data["graph_y"]

random_i = args.dataset.find("random")
if random_i == -1:
    base_name = args.dataset
else:
    base_name = args.dataset[:random_i]

randperm = torch.randperm(graph_y.num_nodes)

for fraction in MASK_FRACTIONS:
    # Replace mask
    n_mask = int(fraction*graph_y.num_nodes)
    unobs_indexes = randperm[:n_mask]

    unobs_mask = torch.zeros(graph_y.num_nodes).to(bool)
    unobs_mask[unobs_indexes] = True
    obs_mask = unobs_mask.bitwise_not()

    graph_y.mask = obs_mask

    # Re-save dataset
    new_name = base_name + "random_{}".format(fraction)

    utils.save_graph_ds(loaded_data, args, new_name)

