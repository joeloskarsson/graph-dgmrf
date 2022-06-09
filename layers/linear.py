import torch
import torch_geometric as ptg
import scipy.linalg as spl

class LinearLayer(ptg.nn.MessagePassing):
    def __init__(self, graph, config):
        super(LinearLayer, self).__init__(aggr="add")

        self.num_nodes = graph.num_nodes
        # in/out degree doesn't matter, undirected graph
        self.degrees = ptg.utils.degree(graph.edge_index[0])

        self.alpha1_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)
        self.alpha2_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)

        if config["use_bias"]:
            self.bias = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)
        else:
            self.bias = None

    @property
    def self_weight(self):
        # Forcing alpha1 to be positive is no restriction on the model
        return torch.exp(self.alpha1_param)

    @property
    def neighbor_weight(self):
        # Second parameter is (alpha2 / alpha1)
        return self.self_weight * torch.tanh(self.alpha2_param)

    def weight_self_representation(self, x):
        # Representation of same node weighted with degree
        return (x.view(-1,self.num_nodes) * self.degrees).view(-1,1)

    def forward(self, x, edge_index, transpose, with_bias):
        weighted_repr = self.weight_self_representation(x)

        aggr = (self.self_weight * weighted_repr) + (self.neighbor_weight*self.propagate(
            edge_index, x=x, transpose=transpose)) # Shape (n_nodes*n_graphs,1)

        if self.bias and with_bias:
            aggr += self.bias

        return aggr

    def log_det(self):
        raise NotImplementedError("log_det not implemented for layer")

