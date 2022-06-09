import numpy as np
import networkx as nx
import torch_geometric as ptg
import scipy.sparse as sps
import tensorflow as tf
import gpflow
import os
import pickle

from graph_matern.kernels.graph_matern_kernel import GraphMaternKernel
from graph_matern.svgp import GraphSVGP

import utils
import constants

LR = 1e-3
N_ITER = 250000
PRINT_INTERVAL = 500
PATIENCE = 5
BATCH_SIZE = 128

@tf.function
def opt_step(opt, loss, variables):
    opt.minimize(loss, var_list=variables)

def graph_gp(X_train, y_train, X_test, graph, config, **kwargs):
    # Standardize target
    graph = graph.to("cpu")
    y = graph.x.squeeze().numpy().astype(np.float64)
    y_std = y.std()
    y_mean = y.mean()
    y_standard = (y - y_mean)/y_std

    # Pre-process data into correct formats
    train_mask = graph.mask.numpy()
    test_mask = np.logical_not(train_mask)
    x_index = np.expand_dims(np.arange(graph.num_nodes, dtype=np.float64), axis=1)
    y_standard = np.expand_dims(y_standard, axis=1)

    if utils.is_none(X_train):
        x_train = x_index[train_mask]
        x_test = x_index[test_mask]
    else:
        x_train = np.concatenate((x_index[train_mask], X_train), axis=1)
        x_test = np.concatenate((x_index[test_mask], X_test), axis=1)

    y_train = y_standard[train_mask]
    y_test = y_standard[test_mask]
    data_train = (x_train, y_train)
    n_train = x_train.shape[0]

    # Compute eigenpairs for graph laplacian
    if config["weight_laplacian"]:
        assert hasattr(graph, "edge_attr"), "No edge attributes in dataset"
        edge_attr = ("edge_attr",)
    else:
        edge_attr = None

    graph_nx = ptg.utils.to_networkx(graph, edge_attrs=edge_attr, to_undirected=True)
    laplacian = sps.csr_matrix(nx.laplacian_matrix(graph_nx, weight="edge_attr"),
            dtype=np.float64)

    # Note: Need the smallest eigenvalues for approximation
    eig_file_path = os.path.join(constants.DS_DIR, config["dataset"],
            "eig_pairs_{}.eig".format(config["n_eigpairs"]))
    if os.path.isfile(eig_file_path):
        print("Eigenpairs file found!")
        with open(eig_file_path, "rb") as f:
            eigvects, eigvals = pickle.load(f)
    else:
        print("Eigenpairs file not found, performing computation...")
        eigenvalues, eigenvectors = tf.linalg.eigh(laplacian.toarray())
        eigvects = eigenvectors[:, :config["n_eigpairs"]]
        eigvals = eigenvalues[:config["n_eigpairs"]]

        with open(eig_file_path, "wb") as f:
            pickle.dump((eigvects.numpy(), eigvals.numpy()), f)

    eigvals = tf.convert_to_tensor(eigvals, dtype=tf.float64)
    eigvects = tf.convert_to_tensor(eigvects, dtype=tf.float64)

    # Create kernel and GP
    vertex_dim = x_train.shape[1] - 1
    if vertex_dim > 0:
        # Use ARD (separate lengthscales for different dims)
        point_kernel = gpflow.kernels.Matern32(lengthscales=[1.]*vertex_dim)
    else:
        point_kernel = None

    kernel = GraphMaternKernel((eigvects, eigvals), vertex_dim=vertex_dim,
            point_kernel=point_kernel, dtype=tf.float64)

    # Train hyperparameters (mainly taken directly from example code)
    @tf.function
    def opt_step(opt, loss, variables):
        opt.minimize(loss, var_list=variables)

    def optimize_SVGP(model, optimizers, steps, q_diag=True):
        if not q_diag:
            gpflow.set_trainable(model.q_mu, False)
            gpflow.set_trainable(model.q_sqrt, False)

        adam_opt, natgrad_opt = optimizers

        variational_params = [(model.q_mu, model.q_sqrt)]

        autotune = tf.data.experimental.AUTOTUNE
        data_minibatch = (
            tf.data.Dataset.from_tensor_slices(data_train)
                .prefetch(autotune)
                .repeat()
                .shuffle(n_train)
                .batch(BATCH_SIZE)
        )
        data_minibatch_it = iter(data_minibatch)
        loss = model.training_loss_closure(data_minibatch_it)
        adam_params = model.trainable_variables
        natgrad_params = variational_params

        adam_opt.minimize(loss, var_list=adam_params)
        if not q_diag:
            natgrad_opt.minimize(loss, var_list=natgrad_params)

        best_elbo = -1.*np.inf
        best_elbo_i = 0
        for step in range(steps):
            opt_step(adam_opt, loss, adam_params)
            if not q_diag:
                opt_step(natgrad_opt, loss, natgrad_params)
            if step % PRINT_INTERVAL == 0:
                likelihood = model.elbo(data_train).numpy()
                print('i={}| ELBO = {:.5}'.format(step, likelihood))

                if likelihood > best_elbo:
                    best_elbo = likelihood
                    best_elbo_i = step

                if (step - best_elbo_i) >= PATIENCE*PRINT_INTERVAL:
                    # Wait for patience prints for improvement
                    break

    gauss_like = gpflow.likelihoods.Gaussian(variance=1e-4)
    model = GraphSVGP(
        kernel=kernel,
        likelihood=gauss_like, # Note: changed from example
        inducing_variable=[0]*config["n_eigpairs"],
        num_latent_gps=1,
        whiten=True,
        q_diag=True,
    )
    adam_opt = tf.optimizers.Adam(0.001)
    natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.001)

    optimize_SVGP(model, (adam_opt, natgrad_opt), N_ITER, True)
    gpflow.utilities.print_summary(model)

    # Predict on test data
    gp_mean, gp_var = model.predict_y(x_test) # Note: y, not f

    pred_mean = gp_mean.numpy().squeeze(1)
    pred_std = np.sqrt(gp_var.numpy().squeeze(1))

    # Un-standardize
    pred_mean = (pred_mean * y_std) + y_mean
    pred_std = pred_std * y_std
    return pred_mean, pred_std

