import numpy as np
import torch_geometric as ptg
import scipy.sparse as sparse
from sksparse.cholmod import cholesky
import scipy.stats as sps

import utils

def gmrf_pdf(x, mu, Q, factor=None):
    diff = x-mu
    a = diff.T@(Q@diff)

    if utils.is_none(factor):
        factor = cholesky(Q)
    log_det = factor.logdet() # Note: log-det of Q, not L

    return 0.5*log_det - 0.5*a

def igmrf(X_train, y_train, X_test, graph, config, **kwargs):
    eps = config["igmrf_eps"]
    N = graph.num_nodes
    graph = graph.to("cpu")

    # Make data zero mean
    y = graph.x.squeeze().numpy()
    data_mean = np.mean(y)
    y = y - data_mean

    # Construct prior Q (0 mean prior)
    A = ptg.utils.to_scipy_sparse_matrix(graph.edge_index)
    A = sparse.csc_matrix(A)
    degrees = ptg.utils.degree(graph.edge_index[0]).numpy() # Note: undirected graph
    D = sparse.diags(degrees, offsets=0, format="csc")

    Q_unweighted = D - A
    mask = graph.mask.numpy()
    float_mask = mask.astype(float)
    y_m = float_mask * y

    # Tune parameter kappa
    # Use trick to compute marginal likelihood p(y|kappa) using arbitrary x
    x = np.zeros(N)
    sigmas = np.logspace(-3, 0, 4)
    kappas = np.logspace(-2, 3, 20)
    param_space = np.array(np.meshgrid(sigmas, kappas)).T.reshape(-1,2)
    print("Tuning parameters kappa, sigma ...")

    def comp_dist(kappa, sigma):
        Q = kappa*Q_unweighted + eps*sparse.eye(N, format="csc")

        # Posterior
        Q_tilde = Q + (sigma**(-2))*sparse.diags(float_mask, offsets=0, format="csc")
        factor_tilde = cholesky(Q_tilde)
        mu_tilde = factor_tilde((sigma**(-2))*y_m) # Solve for Q_tilde

        return Q, Q_tilde, factor_tilde, mu_tilde

    likelihoods = np.zeros(param_space.shape[0])
    for p_i, (s,k) in enumerate(param_space):
        log_like = np.sum(sps.norm.logpdf(y_m, loc=x, scale=s)[mask])
        Q, Q_tilde, factor_tilde, mu_tilde = comp_dist(k, s)

        log_prior = gmrf_pdf(x, np.zeros_like(x), Q)
        log_post = gmrf_pdf(x, mu_tilde, Q_tilde, factor=factor_tilde)

        marg_like = log_like + log_prior - log_post
        likelihoods[p_i] = marg_like

        print("k={}, s={}| marginal likelihood: {:.6}".format(k, s, marg_like))

    # Compute predictions for best kappa
    likelihoods[np.isnan(likelihoods)] = -np.inf
    best_s, best_k = param_space[np.argmax(likelihoods)]
    print("Best k: {}".format(best_k))
    print("Best s: {}".format(best_s))

    _, _, factor_tilde, mu_tilde = comp_dist(best_k, best_s)
    post_mean = mu_tilde + data_mean # Add back data_mean to predictions

    # Sample and take MC variance estimate
    def sample_post():
        z = np.random.randn(N)
        sample = mu_tilde + factor_tilde.apply_Pt(
                factor_tilde.solve_Lt(z, use_LDLt_decomposition=False)) # Must permute
        return sample

    post_samples = np.stack([sample_post() for s_i in range(config["post_samples"])],
            axis=0)
    post_var_x = np.mean((post_samples - mu_tilde)**2, axis=0)

    post_std_y = np.sqrt(post_var_x + best_s**2)

    pred_mean = post_mean[~mask]
    pred_std = post_std_y[~mask]

    return pred_mean, pred_std

