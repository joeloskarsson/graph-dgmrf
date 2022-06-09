import torch
import torch_geometric as ptg
import numpy as np
import networkx as nx
import json
import os
import time
import pickle
import argparse
import wandb
import copy

from lib.cg_batch import cg_batch
import visualization as vis
import vi
from dgmrf import DGMRF
import constants
import utils
import inference

def get_config():
    parser = argparse.ArgumentParser(description='Graph DGMRF')
    # If config file should be used
    parser.add_argument("--config", type=str, help="Config file to read run config from")

    # General
    parser.add_argument("--dataset", type=str, default="toy_gmrf42_random",
            help="Which dataset to use")
    parser.add_argument("--seed", type=int, default=123,
            help="Seed for random number generator")
    parser.add_argument("--noise_std", type=int, default=1e-2,
            help="Value to use for noise std.-dev. (if not learned, otherwise initial)")
    parser.add_argument("--learn_noise_std", type=int, default=1,
            help="If the noise std.-dev. should be learned jointly with the model")
    parser.add_argument("--optimizer", type=str, default="adam",
            help="Optimizer to use for training")
    parser.add_argument("--print_params", type=int, default=0,
            help="Write out parameter values during training")
    parser.add_argument("--features", type=int, default=0,
            help="Include additional node-features, apart from Gaussian field")
    parser.add_argument("--coeff_inv_std", type=float, default=0.0001,
            help="Inverse standard deviation of coefficients beta (feature weights)")

    # Model Architecture
    parser.add_argument("--n_layers", type=int,
                        help="Number of message passing layers", default=1)
    parser.add_argument("--use_bias", type=int, default=1,
                        help="Use bias parameter in layers")
    parser.add_argument("--non_linear", type=int, default=0,
                        help="Add in non-linear layers, requiring VI for posterior")
    parser.add_argument("--dist_weight", type=int, default=0,
                        help="Use distance weighted adjacency matrix")
    parser.add_argument("--fix_gamma", type=int, default=0,
                        help="If the value of the gamma parameter should be fixed")
    parser.add_argument("--gamma_value", type=float, default=1.0,
                        help="Value for gamma when fixed")

    # Training
    parser.add_argument("--log_det_method", type=str, default="eigvals",
        help="Method for log-det. computations (eigvals/dad), dad is using power series")
    parser.add_argument("--n_iterations", type=int,
            help="How many iterations to train for", default=1000)
    parser.add_argument("--val_interval", type=int, default=100,
            help="Evaluate model every val_interval:th iteration")
    parser.add_argument("--n_training_samples", type=int, default=10,
        help="Number of samples to use for each iteration in training")
    parser.add_argument("--lr", type=float,
            help="Learning rate", default=0.01)
    parser.add_argument("--vi_layers", type=int, default=0,
        help="Flex-layers to apply to independent vi-samples to introduce correlation")

    # Posterior inference
    parser.add_argument("--n_post_samples", type=int, default=100,
        help="Number of samples to draw from posterior for MC-estimate of std.-dev.")
    parser.add_argument("--vi_eval", type=int, default=0,
        help="Use variational distribution in place of true posterior, in evaluation")
    parser.add_argument("--inference_rtol", type=float, default=1e-7,
            help="rtol for CG during inference")

    # Plotting
    parser.add_argument("--plot_vi_samples", type=int, default=3,
        help="Number of vi samples to plot")
    parser.add_argument("--plot_post_samples", type=int, default=3,
        help="Number of posterior samples to plot")
    parser.add_argument("--save_pdf", type=int, default=1,
        help="If plots should also be saved as .pdf-files")
    parser.add_argument("--dump_prediction", type=int, default=0,
        help="If produced graphs should be saved to files")

    args = parser.parse_args()
    config = vars(args)

    # Read additional config from file
    if args.config:
        assert os.path.exists(args.config), "No config file: {}".format(args.config)
        with open(args.config) as json_file:
            config_from_file = json.load(json_file)

        # Make sure all options in config file also exist in argparse config.
        # Avoids choosing wrong parameters because of typos etc.
        unknown_options = set(config_from_file.keys()).difference(set(config.keys()))
        unknown_error = "\n".join(["Unknown option in config file: {}".format(opt)
            for opt in unknown_options])
        assert (not unknown_options), unknown_error

        config.update(config_from_file)

    return config

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    config = get_config()

    # (implementation details force this, but not a problem)
    assert config["plot_vi_samples"] <= config["n_training_samples"], (
            "plot_vi_samples must be less or equal to than n_training_samples")

    # Set all random seeds
    seed_all(config["seed"])

    # Device setup
    if torch.cuda.is_available():
        # Make all tensors created go to GPU
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        # For reproducability on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    dataset_dict = utils.load_dataset(config["dataset"])
    graph_y = dataset_dict["graph_y"]

    if config["features"]:
        assert hasattr(graph_y, "features"), "No features found for dataset"

    # Init wandb
    wandb_name = "{}-{}".format(config["dataset"],
                time.strftime("%H-%M"))
    wandb.init(project=constants.WANDB_PROJECT, config=config, name=wandb_name)

    # Define DGMRF
    dgmrf = DGMRF(graph_y, config)
    opt_params = tuple(dgmrf.parameters())

    val_mask = torch.logical_not(graph_y.mask) # True for unobserved nodes
    graph_y.n_observed = torch.sum(graph_y.mask).to(torch.float32)
    graph_y.n_unobserved = torch.sum(val_mask).to(torch.float32)

    config["log_noise_std"] = torch.log(torch.tensor(config["noise_std"]))
    if config["learn_noise_std"]:
        # Initalize using noise_std in config
        config["log_noise_std"] = torch.nn.parameter.Parameter(config["log_noise_std"])
        opt_params += (config["log_noise_std"],)

    # Train using VI
    vi_dist = vi.VariationalDist(config, graph_y)
    opt_params += tuple(vi_dist.parameters())

    opt = utils.get_optimizer(config["optimizer"])(opt_params, lr=config["lr"])
    total_loss = torch.zeros(1)

    # Training loop
    best_loss = None
    best_params = None

    for iteration_i in range(config["n_iterations"]):
        opt.zero_grad()

        loss = vi.vi_loss(dgmrf, vi_dist, graph_y, config)

        # Train
        loss.backward()
        opt.step()

        total_loss += loss.detach()

        if ((iteration_i+1) % config["val_interval"]) == 0:
            # Validate
            val_samples = vi_dist.sample()

            if config["features"]:
                # Just sample coefficients again (independent of vi x samples)
                vi_coeff_samples = vi_dist.sample_coeff(config["n_training_samples"])
                val_samples = val_samples +\
                    vi_coeff_samples@graph_y.features.transpose(0,1)

            val_error = (1./(config["n_training_samples"]*graph_y.n_unobserved))*\
                    torch.sum(torch.pow(
                        (val_samples - graph_y.x.flatten()), 2)[:, val_mask])
            val_error = val_error.item()

            mean_loss = (total_loss.item() / config["val_interval"])
            total_loss = torch.zeros(1)

            print("Iteration: {}, loss: {:.6}, val_error: {}".format(
                (iteration_i+1), mean_loss, val_error))

            wandb.log({
                "loss": mean_loss,
                "val_error": val_error,
                "iteration": (iteration_i+1),
                "noise_std": utils.noise_std(config),
            })

            # Save best params
            if (best_loss == None) or (mean_loss < best_loss):
                best_params = copy.deepcopy(dgmrf.state_dict())
                best_loss = mean_loss

            if config["print_params"]:
                utils.print_params(dgmrf, config, "--- Model parameters ---")

    # Reload best parameters
    dgmrf.load_state_dict(best_params)
    utils.print_params(dgmrf, config, "Final parameters:")

    # Plot y
    vis.plot_graph(graph_y, name="y", title="y")

    # Plot VI
    vi_evaluation = config["vi_eval"] or config["non_linear"]
    if config["plot_vi_samples"]:
        graph_vi_sample = utils.new_graph(graph_y)
        vi_samples = vi_dist.sample()[:config["plot_vi_samples"]]
        # VI samples plotted when using features are only x (without linear model)
        for sample_i, vi_sample in enumerate(vi_samples.detach()):
            graph_vi_sample.x = vi_sample.unsqueeze(1)
            vis.plot_graph(graph_vi_sample, name="vi_sample",
                    title="VI Sample {}".format(sample_i))
    if not vi_evaluation:
        graph_vi_mean, graph_vi_std = vi_dist.posterior_estimate(graph_y, config)
        vis.plot_graph(graph_vi_mean, name="vi_mean", title="VI Mean")
        vis.plot_graph(graph_vi_std, name="vi_std_dev", title="VI Std-dev.")

    # Posterior inference
    if hasattr(graph_y, "adj_matrix"):
        # Make sure to delete stored adjacency matrix before inference to save memory
        del graph_y.adj_matrix

    # These posteriors are over y
    if vi_evaluation:
        # Use variational distribution in place of true posterior
        print("Using variational distribution as posterior estimate ...")
        graph_post_mean, graph_post_std = vi_dist.posterior_estimate(graph_y, config)
    else:
        # Exact posterior inference
        print("Running posterior inference ...")
        graph_post_mean, graph_post_std = inference.posterior_inference(dgmrf,
                graph_y, config)

    # Plot posterior
    vis.plot_graph(graph_post_mean, name="post_mean", title="Posterior Mean")
    vis.plot_graph(graph_post_std, name="post_std",
            title="Posterior Marginal Std.-Dev.")

    # Compute Metrics
    inverse_mask = torch.logical_not(graph_y.mask)
    if ("graph_post_true_mean" in dataset_dict) and (
            "graph_post_true_std" in dataset_dict):
        graph_post_true_mean = dataset_dict["graph_post_true_mean"]
        graph_post_true_std = dataset_dict["graph_post_true_std"]

        # Plot true posterior
        vis.plot_graph(graph_post_true_mean, name="post_true_mean",
                title="True Posterior Mean")
        vis.plot_graph(graph_post_true_std, name="post_true_std",
                title="True Posterior Std.-dev.")

        mean_diff = (graph_post_mean.x - graph_post_true_mean.x)[inverse_mask, :]
        std_diff = (graph_post_std.x - graph_post_true_std.x)[inverse_mask, :]

        mean_mae = torch.mean(torch.abs(mean_diff))
        std_mae = torch.mean(torch.abs(std_diff))

        print("MAE of posterior mean: {:.7}".format(mean_mae))
        print("MAE of posterior std.-dev.: {:.7}".format(std_mae))
        wandb.run.summary["mean_mae"] = mean_mae
        wandb.run.summary["std_mae"] = std_mae

    # Compare posterior mean with y
    diff = (graph_post_mean.x - graph_y.x)[inverse_mask, :]
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))

    pred_mean_np = graph_post_mean.x[inverse_mask, :].cpu().numpy()
    pred_std_np = graph_post_std.x[inverse_mask, :].cpu().numpy()
    target_np = graph_y.x[inverse_mask, :].cpu().numpy()

    crps =  utils.crps_score(pred_mean_np, pred_std_np, target_np)
    int_score = utils.int_score(pred_mean_np, pred_std_np, target_np)

    print("MAE:  \t{:.7}".format(mae))
    print("RMSE: \t{:.7}".format(rmse))
    print("CRPS: \t{:.7}".format(crps))
    print("INT:  \t{:.7}".format(int_score))
    wandb.run.summary["mae"] = mae
    wandb.run.summary["rmse"] = rmse
    wandb.run.summary["crps"] = crps
    wandb.run.summary["int_score"] = int_score

    if "graph_x" in dataset_dict:
        # Plot x, if known
        graph_x = dataset_dict["graph_x"]
        vis.plot_graph(graph_x, name="x", title="x")

    # Plot additional zooms for dataset
    zoom_list = utils.get_dataset_zooms(config["dataset"])
    for zoom_i, zoom in enumerate(zoom_list):
        # Plot y, posterior mean and posterior std for zooms
        vis.plot_graph(graph_y, name="y", title="y (zoom {})".format(zoom_i), zoom=zoom)
        vis.plot_graph(graph_post_mean, name="post_mean",
                title="Posterior Mean (zoom {})".format(zoom_i), zoom=zoom)
        vis.plot_graph(graph_post_std, name="post_std",
            title="Posterior Marginal Std.-Dev. (zoom {})".format(zoom_i), zoom=zoom)

    # Optionally save prediction graphs
    if config["dump_prediction"]:
        save_graphs = {"post_mean": graph_post_mean, "post_std": graph_post_std}
        if not vi_evaluation:
            save_graphs.update({"vi_mean": graph_vi_mean, "vi_std": graph_vi_std})

        for name, graph in save_graphs.items():
            utils.save_graph(graph, "{}_graph".format(name), wandb.run.dir)

if __name__ == "__main__":
    main()

