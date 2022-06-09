import torch
import torch_geometric as ptg
import scipy.linalg as spl

from layers.linear import LinearLayer
import utils

# Layer: G = D^(gammma)(a1 I + a2 D^(-1)A)
class FlexLayer(LinearLayer):
    def __init__(self, graph, config, vi_layer=False):
        super(FlexLayer, self).__init__(graph, config)

        self.dist_weighted = bool(config["dist_weight"])
        self.eigvals_log_det = bool(config["log_det_method"] == "eigvals")

        if self.dist_weighted:
            if config["log_det_method"] == "eigvals":
                assert hasattr(graph, "weighted_eigvals"), (
                    "Dataset not pre-processed with weighted eigenvalues")
                self.adj_eigvals = graph.weighted_eigvals
            elif config["log_det_method"] == "dad":
                assert hasattr(graph, "weighted_dad_traces"), (
                    "Dataset not pre-processed with DAD traces")
                dad_traces = graph.weighted_dad_traces
            else:
                assert False, "Unknown log-det method"

            self.degrees = graph.weighted_degrees
            self.dist_edge_weights = graph.edge_attr[:,0]
        else:
            if config["log_det_method"] == "eigvals":
                assert hasattr(graph, "eigvals"), (
                    "Dataset not pre-processed with eigenvalues")
                self.adj_eigvals = graph.eigvals
            elif config["log_det_method"] == "dad":
                assert hasattr(graph, "dad_traces"), (
                    "Dataset not pre-processed with DAD traces")
                dad_traces = graph.dad_traces
            else:
                assert False, "Unknown log-det method"

        if config["log_det_method"] == "dad":
            # Complete vector to use in power series for log-det-computation
            k_max = len(dad_traces)
            self.power_ks = torch.arange(k_max) + 1
            self.power_series_vec = (dad_traces * torch.pow(-1., (self.power_ks+1))
                    ) / self.power_ks

        self.log_degrees = torch.log(self.degrees)
        self.sum_log_degrees = torch.sum(self.log_degrees) # For determinant

        # Degree weighting parameter (can not be fixed for vi)
        self.fixed_gamma = (not vi_layer) and bool(config["fix_gamma"])
        if self.fixed_gamma:
            self.gamma_param = config["gamma_value"]*torch.ones(1)
        else:
            self.gamma_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)

        # edge_log_degrees contains log(d_i) of the target node of each edge
        self.edge_log_degrees = self.log_degrees[graph.edge_index[1]]
        self.edge_log_degrees_transpose = self.log_degrees[graph.edge_index[0]]

    @property
    def degree_power(self):
        if self.fixed_gamma:
            return self.gamma_param
        else:
            # Forcing gamma to be in (0,1)
            return torch.sigmoid(self.gamma_param)

    def log_det(self):
        if self.eigvals_log_det:
            # Eigenvalue-based method
            eigvals = self.neighbor_weight[0]*self.adj_eigvals + self.self_weight[0]
            agg_contrib = torch.sum(torch.log(torch.abs(eigvals))) # from (aI+aD^-1A)
            degree_contrib = self.degree_power*self.sum_log_degrees # From D^gamma
            return agg_contrib + degree_contrib
        else:
            # Power series method, using DAD traces
            alpha_contrib = self.num_nodes*self.alpha1_param
            gamma_contrib = self.degree_power*self.sum_log_degrees
            dad_contrib = torch.sum(self.power_series_vec *\
                    torch.pow(torch.tanh(self.alpha2_param), self.power_ks))
            return alpha_contrib + gamma_contrib + dad_contrib

    def weight_self_representation(self, x):
        # Representation of same node weighted with degree (taken to power)
        return (x.view(-1,self.num_nodes)*torch.exp(
            self.degree_power * self.log_degrees)).view(-1,1)

    def message(self, x_j, transpose):
        # x_j are neighbor features
        if transpose:
            log_degrees = self.edge_log_degrees_transpose
        else:
            log_degrees = self.edge_log_degrees

        edge_weights = torch.exp((self.degree_power - 1) * log_degrees)
        if self.dist_weighted:
            edge_weights = edge_weights*self.dist_edge_weights

        weighted_messages = x_j.view(-1, edge_weights.shape[0]) * edge_weights

        return weighted_messages.view(-1,1)

