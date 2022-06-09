import torch
import torch_geometric as ptg

from layers.dad_layer import DADLayer

# Implicitly applies the weighted matrix D^(-1/2)AD^(-1/2) to a batch
class WeightedDADLayer(DADLayer):
    def __init__(self, graph):
        super().__init__(graph)

        self.sqrt_degrees = torch.sqrt(graph.weighted_degrees)
        self.dist_edge_weights = graph.edge_attr[:,0]

    def message(self, x_j):
        # x_j are neighbor features
        edge_weights = self.dist_edge_weights
        weighted_messages = x_j.view(-1, edge_weights.shape[0]) * edge_weights
        return weighted_messages.view(-1,1)

