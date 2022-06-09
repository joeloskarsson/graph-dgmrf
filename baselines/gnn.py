import torch
import torch_geometric as ptg
import numpy as np
import copy

LOSS = torch.nn.MSELoss(reduction='none')
LR = 1e-3
MAX_EPOCHS = int(1e5)
PATIENCE = 100 # Waiting time for E.S.
VAL_INTERVAL = 10
VAL_FRACTION = 0.2

# Hyperparameter configs to try
# (n_layers, hidden_dim)
HYPER_CONFIGS = (
    (1, 32), # 0
    (1, 64),
    (3, 32),
    (3, 64), # 3
    (3, 128),
    (5, 32),
    (5, 64), # 6
    (5, 128),
    (7, 128), # 8
)

LAYERS = {
    "gcn": ptg.nn.GCNConv,
    "gat": ptg.nn.GATConv,
}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GNN(torch.nn.Module):
    def __init__(self, layer_type, in_dim, hidden_dim, n_layers):
        super().__init__()

        assert layer_type in LAYERS, "Unknown GNN layer"
        self.layer_class = LAYERS[layer_type]

        layer_list = []
        layer_list.append(self.layer_class(in_dim, hidden_dim))

        # If more layers
        for _ in range(1,n_layers):
            layer_list.append(torch.nn.ReLU())
            layer_list.append(self.layer_class(hidden_dim, hidden_dim))

        layer_list.append(torch.nn.ReLU())
        layer_list.append(torch.nn.Linear(hidden_dim, 1))
        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        for layer in  self.layers:
            if type(layer) == self.layer_class:
                x = layer(x, edge_index)
            else:
                x = layer(x)

        return x

def eval_gnn(gnn, graph, mask):
    pred = gnn(graph)
    return torch.mean(LOSS(pred, graph.y).squeeze(1)[mask])

def train_gnn(gnn, graph, train_mask, val_mask):
    opt_params = gnn.parameters()
    opt = torch.optim.Adam(opt_params, lr=LR)

    # For early stopping
    best_val_loss = torch.tensor(np.inf)
    best_val_epoch = 0
    best_parameters = None

    for epoch_i in range(1, MAX_EPOCHS+1):
        opt.zero_grad()

        loss = eval_gnn(gnn, graph, train_mask)

        loss.backward()
        opt.step()

        if (epoch_i % VAL_INTERVAL) == 0:
            with torch.no_grad():
                val_loss = eval_gnn(gnn, graph, val_mask)

            print("Epoch {} | train_loss={}, val_loss={}".format(epoch_i,
                loss.item(), val_loss.item()))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch_i
                best_parameters = copy.deepcopy(gnn.state_dict())

            # If PATIENCE epochs without improvement
            if (epoch_i - best_val_epoch) >= PATIENCE:
                break # Early stopping

    # Reload best parameters
    gnn.load_state_dict(best_parameters)

    return gnn, best_val_loss

def gnn(X_train, y_train, X_test, graph, config, X, **kwargs):
    # Device setup
    if torch.cuda.is_available():
        # Make all tensors created go to GPU
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        # For reproducability on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    graph.y = graph.x # graph target is y
    graph.x = torch.tensor(X) # Put together all features in graph
    input_dim = X.shape[1]

    # Create train, val test masks
    ntest_i = torch.arange(graph.num_nodes)[graph.mask] # "non-test"
    n_ntest = ntest_i.shape[0]
    ntest_i_perm = ntest_i[torch.randperm(n_ntest)]
    n_val = int(n_ntest*VAL_FRACTION)
    val_i = ntest_i_perm[:n_val]
    train_i = ntest_i_perm[n_val:]

    val_mask = torch.zeros(graph.num_nodes).to(bool)
    train_mask = torch.zeros(graph.num_nodes).to(bool)
    val_mask[val_i] = True
    train_mask[train_i] = True
    test_mask = torch.logical_not(graph.mask)

    if config["gnn_config"] == -1:
        # Tune hyperparameters
        best_val_loss = torch.tensor(np.inf)
        best_hyper = None
        for cur_hyper in HYPER_CONFIGS:
            n_layers, hidden_dim = cur_hyper

            # Instatiate model
            print("GNN: {}, {} layers, {} hidden dim".format(config["gnn_layer"],
                n_layers, hidden_dim))
            gnn = GNN(config["gnn_layer"], input_dim, hidden_dim, n_layers)
            print("{} parameters".format(count_params(gnn)))

            _, val_loss = train_gnn(gnn, graph, train_mask, val_mask)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyper = cur_hyper
    else:
        best_hyper = HYPER_CONFIGS[config["gnn_config"]]

    # Train and evaluate ensemble
    print("Best hyperparameters: {} layers, {} hidden dim".format(*best_hyper))
    print("Training ensemble of {} models using best hyperparameters".format(
        config["n_ensemble"]))

    best_n_layers, best_hidden_dim = best_hyper
    ens_predictions = []

    for model_i in range(config["n_ensemble"]):
        _ = torch.random.manual_seed(config["seed"] + 100 + model_i) # Arbitrary seeds

        gnn = GNN(config["gnn_layer"], input_dim, best_hidden_dim, best_n_layers)
        gnn, _ = train_gnn(gnn, graph, train_mask, val_mask)

        with torch.no_grad():
            ens_predictions.append(gnn(graph).squeeze(1))

    test_predictions = torch.stack(ens_predictions, axis=0)[:,test_mask]
    pred_mean = torch.mean(test_predictions, axis=0).cpu().numpy()

    print("{} with {} layers, {} hidden dim".format(config["gnn_layer"], *best_hyper))

    if config["n_ensemble"] == 1:
        # Only single model
        return pred_mean, None

    # Ensemble
    pred_std = torch.std(test_predictions, axis=0).cpu().numpy()
    return pred_mean, pred_std

