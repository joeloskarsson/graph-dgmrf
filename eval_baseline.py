import numpy as np
import torch
import argparse
import sklearn.metrics as metrics

import utils

from baselines.bayes_linreg import bayes_linreg
from baselines.mlp import mlp
from baselines.lp import lp
from baselines.igmrf import igmrf
from baselines.gnn import gnn
from baselines.dgp import dgp
from baselines.graph_gp import graph_gp
from baselines.gp import gp

baselines = {
    "bayeslinreg": bayes_linreg,
    "mlp": mlp,
    "lp": lp,
    "igmrf": igmrf,
    "gnn": gnn,
    "dgp": dgp,
    "graphgp": graph_gp,
    "gp": gp,
}

def main():
    # Argparse
    parser = argparse.ArgumentParser(description='Train and evaluate baseline models')

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="toy_gmrf42_random",
            help="Dataset to apply model to")
    parser.add_argument("--model", type=str, default="linreg",
            help="Baseline model to use")
    parser.add_argument("--pos", type=int, default=1,
            help="Include position as feature")
    parser.add_argument("--features", type=int, default=1,
            help="Include additional features")
    parser.add_argument("--plot", type=int, default=0,
            help="Show plots")
    parser.add_argument("--n_ensemble", type=int, default=10,
            help="Amount of GNN models to use in ensemble")

    # --- Model specific ---
    # IGMRF
    parser.add_argument("--igmrf_eps", type=float, default=1e-6,
            help="Epsilon to be added to IGMRF prior diagonal")
    parser.add_argument("--post_samples", type=int, default=100,
            help="Posterior samples for MC estimate in IGMRF")

    # GNN/DGP
    parser.add_argument("--gnn_layer", type=str, default="gcn",
            help="GNN layer type to use")
    parser.add_argument("--gnn_config", type=int, default=-1,
            help="Index of GNN architecture to use (-1 to tune)")

    # Graph-Matern-GP
    parser.add_argument("--n_eigpairs", type=int, default=500,
            help="Amount of eigen-pairs to use for inducing points")
    parser.add_argument("--weight_laplacian", type=int, default=0,
            help="Use edge features to weight graph laplacian")


    config = vars(parser.parse_args())

    assert config["model"] in baselines, "Unknown model"

    # Set seed
    _ = torch.random.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Load data
    dataset_dict = utils.load_dataset(config["dataset"])
    graph_y = dataset_dict["graph_y"]

    y = graph_y.x.cpu().numpy()[:,0] # Sklearn works with 1d y

    feature_mats = []
    if config["pos"]:
        feature_mats.append(graph_y.pos.cpu().numpy())
    if config["features"]:
        feature_mats.append(graph_y.features.cpu().numpy())

    # Split into train and test
    train_mask = graph_y.mask.cpu().numpy()
    test_mask = np.logical_not(train_mask)
    y_train = y[train_mask]
    y_test = y[test_mask]

    if not feature_mats:
        X, X_train, X_test = None, None, None
    else:
        X = np.concatenate(feature_mats, axis=1)
        X_train = X[train_mask]
        X_test = X[test_mask]

    pred_mean, pred_std = baselines[config["model"]](X_train, y_train, X_test,
            graph=graph_y, config=config, X=X)

    test_mae = metrics.mean_absolute_error(y_test, pred_mean)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, pred_mean))

    if utils.is_none(pred_std):
        test_crps = "-"
        test_int = "-"
    else:
        test_crps = utils.crps_score(pred_mean, pred_std, y_test)
        test_int = utils.int_score(pred_mean, pred_std, y_test)

    print("Model: {}".format(config["model"]))
    print("MAE:  \t{:.7}".format(test_mae))
    print("RMSE: \t{:.7}".format(test_rmse))
    print("CRPS: \t{:.7}".format(test_crps))
    print("INT:  \t{:.7}".format(test_int))
    print("{:.7},{:.7},{:.7},{:.7}".format(test_mae, test_rmse,
        test_crps, test_int))


if __name__ == "__main__":
    main()

