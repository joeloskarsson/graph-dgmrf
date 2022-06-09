import torch
import torch_geometric as ptg

import utils
from layers.flex import FlexLayer

class VariationalDist(torch.nn.Module):
    def __init__(self, config, graph_y):
        super().__init__()

        # Dimensionality of distribution (num_nodes of graph)
        self.dim = graph_y.num_nodes

        # Standard amount of samples (must be fixed to be efficient)
        self.n_samples = config["n_training_samples"]

        # Variational distribution, Initialize with observed y
        self.mean_param = torch.nn.parameter.Parameter(graph_y.mask*graph_y.x[:,0])
        self.diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(self.dim) - 1.) # U(-1,1)

        self.layers = torch.nn.ModuleList([FlexLayer(graph_y, config, vi_layer=True)
                                           for _ in range(config["vi_layers"])])
        if config["vi_layers"] > 0:
            self.post_diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(self.dim) - 1.)

        # Reuse same batch with different x-values
        self.sample_batch = ptg.data.Batch.from_data_list([utils.new_graph(graph_y)
                                    for _ in range(self.n_samples)])

        if config["features"]:
            # Additional variational distribution for linear coefficients
            n_features = graph_y.features.shape[1]
            self.coeff_mean_param = torch.nn.parameter.Parameter(torch.randn(n_features))
            self.coeff_diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(n_features) - 1.) # U(-1,1)

            self.coeff_inv_std = config["coeff_inv_std"]

    @property
    def std(self):
        # Note: Only std before layers
        return torch.nn.functional.softplus(self.diag_param)

    @property
    def post_diag(self):
        # Diagonal of diagonal matrix applied after layers
        return torch.nn.functional.softplus(self.post_diag_param)

    @property
    def coeff_std(self):
        return torch.nn.functional.softplus(self.coeff_diag_param)

    def sample(self):
        standard_sample = torch.randn(self.n_samples, self.dim)
        ind_samples = self.std * standard_sample

        self.sample_batch.x = ind_samples.reshape(-1,1) # Stack all
        for layer in self.layers:
            propagated = layer(self.sample_batch.x, self.sample_batch.edge_index,
                    transpose=False, with_bias=False)
            self.sample_batch.x = propagated

        samples = self.sample_batch.x.reshape(self.n_samples, -1)
        if self.layers:
            # Apply post diagonal matrix
            samples  = self.post_diag * samples
        samples = samples + self.mean_param # Add mean last (not changed by layers)
        return samples # shape (n_samples, n_nodes)

    def log_det(self):
        layers_log_det = sum([layer.log_det() for layer in self.layers])
        std_log_det = torch.sum(torch.log(self.std))
        total_log_det = 2.0*std_log_det + 2.0*layers_log_det

        if self.layers:
            post_diag_log_det = torch.sum(torch.log(self.post_diag))
            total_log_det = total_log_det + 2.0*post_diag_log_det

        return total_log_det

    def sample_coeff(self, n_samples):
        standard_sample = torch.randn(n_samples, self.coeff_mean_param.shape[0])
        samples = (self.coeff_std * standard_sample) + self.coeff_mean_param
        return samples # shape (n_samples, n_features)

    def log_det_coeff(self):
        return 2.0*torch.sum(torch.log(self.coeff_std))

    def ce_coeff(self):
        # Compute Cross-entropy term (CE between VI-dist and coeff prior)
        return -0.5*(self.coeff_inv_std**2)*torch.sum(
                torch.pow(self.coeff_std, 2) + torch.pow(self.coeff_mean_param, 2))

    @torch.no_grad()
    def posterior_estimate(self, graph_y, config):
        # Compute mean and marginal std of distribution (posterior estimate)
        # Mean
        graph_post_mean = utils.new_graph(graph_y,
                new_x=self.mean_param.detach().unsqueeze(1))

        # Marginal std. (MC estimate)
        mc_sample_list = []
        cur_mc_samples = 0
        while cur_mc_samples < config["n_post_samples"]:
            mc_sample_list.append(self.sample())
            cur_mc_samples += self.n_samples
        mc_samples = torch.cat(mc_sample_list, dim=0)[:config["n_post_samples"]]

        # MC estimate of variance using known population mean
        post_var_x = torch.mean(torch.pow(mc_samples - self.mean_param, 2), dim=0)
        # Posterior std.-dev. for y
        post_std = torch.sqrt(post_var_x + utils.noise_var(config)).unsqueeze(1)

        graph_post_std = utils.new_graph(graph_y, new_x=post_std)

        return graph_post_mean, graph_post_std


def vi_loss(dgmrf, vi_dist, graph_y, config):
    vi_samples = vi_dist.sample()
    vi_log_det = vi_dist.log_det()
    vi_dist.sample_batch.x = vi_samples.reshape(-1,1)
    # Column vector of node values for all samples

    g = dgmrf(vi_dist.sample_batch) # Shape (n_training_samples*n_nodes, 1)

    # Construct loss
    l1 = 0.5*vi_log_det
    l2 = -graph_y.n_observed*config["log_noise_std"]
    l3 = dgmrf.log_det()
    l4 = -(1./(2. * config["n_training_samples"])) * torch.sum(torch.pow(g,2))

    if config["features"]:
        vi_coeff_samples = vi_dist.sample_coeff(config["n_training_samples"])
        # Mean from a VI sample (x + linear feature model)
        vi_samples = vi_samples + vi_coeff_samples@graph_y.features.transpose(0,1)

        # Added term when using additional features
        vi_coeff_log_det = vi_dist.log_det_coeff()
        entropy_term = 0.5*vi_coeff_log_det
        ce_term = vi_dist.ce_coeff()

        l1 = l1 + entropy_term
        l4 = l4 + ce_term

    l5 = -(1./(2. * utils.noise_var(config)*\
        config["n_training_samples"]))*torch.sum(torch.pow(
            (vi_samples - graph_y.x.flatten()), 2)[:, graph_y.mask])

    elbo = l1 + l2 + l3 + l4 + l5
    loss = (-1./graph_y.num_nodes)*elbo
    return loss

