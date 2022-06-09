import torch
import torch.distributions as dists
import torch_geometric as ptg

from layers.flex import FlexLayer
from layers.activation import DGMRFActivation

class DGMRF(torch.nn.Module):
    def __init__(self, graph, config):
        super(DGMRF, self).__init__()

        layer_list = []
        for layer_i in range(config["n_layers"]):
            layer_list.append(FlexLayer(graph, config))

            # Optionally add non-linearities between hidden layers
            if config["non_linear"] and (layer_i < (config["n_layers"]-1)):
                layer_list.append(DGMRFActivation(config))

        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, data, transpose=False, with_bias=True):
        x = data.x

        if transpose:
            # Transpose operation means reverse layer order
            layer_iter = reversed(self.layers)
        else:
            layer_iter = self.layers

        for layer in layer_iter:
            x = layer(x, data.edge_index, transpose, with_bias)

        return x

    def log_det(self):
        return sum([layer.log_det() for layer in self.layers])

