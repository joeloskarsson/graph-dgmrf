import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import MeanFieldVariationalDistribution
from gpytorch.variational import VariationalStrategy

NUM_EPOCHS = 100

def gp(X_train, y_train, X_test, **kwargs):
    # Closely follows the GPytorch tutorial at https://docs.gpytorch.ai/en/v1.4.2/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html#Creating-a-SVGP-Model
    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device) # (N_data, d_in)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device) # (N_data, d_in)
    y_train = torch.tensor(y_train, dtype=torch.float32,
            device=device) # (N_data, 1)

    # Define data loading
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset = TensorDataset(X_test) # Note, no y
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Define model
    class GPModel(ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = CholeskyVariationalDistribution(
                    inducing_points.shape[0])
            variational_strategy = VariationalStrategy(self, inducing_points,
                    variational_distribution, learn_inducing_locations=True)
            super(GPModel, self).__init__(variational_strategy)

            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5,
                        ard_num_dims=inducing_points.shape[1]))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    ind_point_init_i = torch.randperm(X_train.shape[0])[:500]
    inducing_points = X_train[ind_point_init_i] # Init with 500 inducing points
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.shape[0])

    for i in range(NUM_EPOCHS):
        # Within each iteration, we will go over each minibatch of data
        epoch_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -torch.mean(mll(output, y_batch))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss*x_batch.shape[0])
        epoch_loss = torch.sum(torch.stack(epoch_losses))/y_train.shape[0]
        print(f"Epoch {i} - loss={epoch_loss:.6}")


    # Evaluation
    model.eval()
    likelihood.eval()

    means = []
    stds = []
    with torch.no_grad():
        for x_batch, in test_loader:
            f_preds = model(x_batch)
            y_preds = likelihood(f_preds)

            pred_means = f_preds.mean
            pred_std = torch.sqrt(y_preds.variance)

            means.append(pred_means)
            stds.append(pred_std)

    full_pred_mean = torch.cat(means, dim=0).cpu().numpy()
    full_pred_std = torch.cat(stds, dim=0).cpu().numpy()

    print(f"Noise var. = {likelihood.noise[0]}")

    return full_pred_mean, full_pred_std

