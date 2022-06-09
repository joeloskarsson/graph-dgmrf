import torch
import numpy as np

import utils
import visualization as vis
from layers.neighbor_mean import NeighborMean
from lib.cg_batch import cg_batch

def lp(X_train, y_train, X_test, graph, config, **kwargs):
    test_mask = np.logical_not(graph.mask.cpu().numpy())
    labeled_y = graph.x.clone()
    labeled_y[test_mask, :] = 0. # Set unobserved to 0

    nm_layer = NeighborMean()
    rhs = nm_layer(labeled_y, graph.edge_index)[test_mask,:]

    # Apply matrix (I - P_uu) to a vector
    graph_input = torch.zeros_like(graph.x)
    def apply_mat(v):
        graph_input[test_mask,:] = v

        mean_res = nm_layer(graph_input, graph.edge_index)[test_mask,:]
        return v - mean_res

    apply_mat_batched = (lambda v: apply_mat(v[0]).unsqueeze(0))

    # CG
    solution, cg_info = cg_batch(apply_mat_batched, rhs.unsqueeze(0), rtol=1e-4)
    print("CG finished in {} iterations, solution optimal: {}".format(
        cg_info["niter"], cg_info["optimal"]))

    pred_mean = solution[0,:,0].cpu().numpy()

    if config["plot"]:
        pred_graph = utils.new_graph(graph, new_x=graph.x)
        pred_graph.x[test_mask, 0] = solution[0,:,0]
        vis.plot_graph(pred_graph, "lp_pred", title="LP Prediction", show=True)

    return pred_mean, None

