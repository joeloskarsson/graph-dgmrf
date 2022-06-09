# "Deep graph prior"

import torch
import torch_geometric as ptg
import numpy as np

from baselines.gnn import gnn

def dgp(X_train, y_train, X_test, graph, config, X, **kwargs):
    z = torch.randn(graph.num_nodes, 1)

    return gnn(X_train, y_train, X_test, graph=graph, config=config, X=z.numpy())

