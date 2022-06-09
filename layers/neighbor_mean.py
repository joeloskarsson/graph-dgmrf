import torch
import torch_geometric as ptg
import scipy.linalg as spl

class NeighborMean(ptg.nn.MessagePassing):
    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

