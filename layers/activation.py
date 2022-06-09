import torch

class DGMRFActivation(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight_param = torch.nn.parameter.Parameter(
                2*torch.rand(1) - 1.) # U(-1,1)

        # For log-det
        self.n_training_samples = config["n_training_samples"]
        self.last_input = None

    @property
    def activation_weight(self):
        return torch.nn.functional.softplus(self.weight_param)

    def forward(self, x, edge_index, transpose, with_bias):
        self.last_input = x.detach()
        return torch.nn.functional.prelu(x, self.activation_weight)

    def log_det(self):
        # Computes log-det for last input fed to forward
        n_negative = (self.last_input < 0.).sum().to(torch.float32)
        return (1./self.n_training_samples)*n_negative*torch.log(self.activation_weight)

