import torch
import torch_geometric as ptg

# Implicitly applies the matrix D^(-1/2)AD^(-1/2) to a batch
class DADLayer(ptg.nn.MessagePassing):
    def __init__(self, graph):
        super(DADLayer, self).__init__(aggr="add")

        self.sqrt_degrees = torch.sqrt(ptg.utils.degree(graph.edge_index[0]))
        self.num_nodes = graph.num_nodes

    def forward(self, x, edge_index):
        x = (x.view(-1,self.num_nodes) / self.sqrt_degrees).view(-1,1) # Apply D^(-1/2)
        x = self.propagate(edge_index, x=x) # Apply A
        x = (x.view(-1,self.num_nodes) / self.sqrt_degrees).view(-1,1) # Apply D^(-1/2)

        return x

