import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
sys.path.insert(0,'..')
import utils

# Set font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

LINE_COLOR = "purple"

# Constants
RES_FILE = "synth_res.csv"
OUTPUT_DIR = "figures"

LAYER_DS = "toy_gmrf42"
DENSE_DS = "toy_gmrf31"

LINE_DS = {
    "mix": "gmrf_prec_mix32_random",
    "dense": "toy_gmrf31_3_densified_random",
}

MAX_LAYERS = 5

layer_ds_list = [LAYER_DS + "_random"] +\
    [LAYER_DS + "_{}_layers_random".format(i) for i in range(2,5)]
dense_ds_list = [DENSE_DS + "_random"] +\
    [DENSE_DS + "_{}_densified_random".format(i) for i in range(2,5)]

def get_true_pred(ds_name):
    # Compute prediction metrics using known true posterior
    dataset_dict = utils.load_dataset(ds_name, ds_dir="../dataset")

    graph_y = dataset_dict["graph_y"]
    graph_post_mean = dataset_dict["graph_post_true_mean"]
    graph_post_std = dataset_dict["graph_post_true_std"]

    inverse_mask = torch.logical_not(graph_y.mask)

    diff = (graph_post_mean.x - graph_y.x)[inverse_mask, :]
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))

    pred_mean_np = graph_post_mean.x[inverse_mask, :].cpu().numpy()
    pred_std_np = graph_post_std.x[inverse_mask, :].cpu().numpy()
    target_np = graph_y.x[inverse_mask, :].cpu().numpy()
    crps = utils.crps_score(pred_mean_np, pred_std_np, target_np)
    int_score = utils.int_score(pred_mean_np, pred_std_np, target_np)

    return {
        "mean-mae": 0., # True posterior, no error
        "std-mae": 0.,
        "mae": mae,
        "rmse": rmse,
        "crps": crps,
        "int": int_score,
    }


# Put results in nice format
df = pd.read_csv(RES_FILE)
ds_list = df["dataset"].to_numpy()
layer_list = df["n_layers"].to_numpy()
res_arr = df.to_numpy()[:,3:].astype(float)

all_ds = list(LINE_DS.values()) + layer_ds_list + dense_ds_list
ds_res = {}
for ds in all_ds:
    layer_res = []
    for layers in range(1, MAX_LAYERS+1):
        # Extract rows associated with dataset and layer
        res_rows = res_arr[np.logical_and(ds_list == ds, layer_list == layers)]
        layer_res.append({
            "mean": np.mean(res_rows, axis=0),
            "std": np.std(res_rows, axis=0, ddof=1), # Unbiased estimate
        })
    ds_res[ds] = layer_res

metric_is = {
    "mean_mae": 1,
    "std_mae": 2,
}
plot_hms = ("mean_mae_mean", "std_mae_mean")
plot_titles = (r"$\text{MAE}_{\mu}$", r"$\text{MAE}_{\sigma}$")

# Make heatmaps
for ds_list, title in zip((layer_ds_list, dense_ds_list), ("Layered", "Densified")):
    heatmaps = {}
    for metric_name, metric_i in metric_is.items():
        for stat in ("mean", "std"):
            heatmap = np.zeros((len(ds_list), MAX_LAYERS))
            # First index: true layers
            # Second index: model layers
            for ds_i, ds in enumerate(ds_list):
                heatmap[ds_i, :] = [res[stat][metric_i] for res in ds_res[ds]]

                heatmap[ds_i, :] = heatmap[ds_i, :] / heatmap[ds_i, ds_i]

            # Normalize somehow?
            #heatmap = heatmap / np.min(heatmap, axis=1, keepdims=True)

            heatmaps[metric_name + "_" + stat] = heatmap

    vmin = min(np.min(heatmaps[hm_name]) for hm_name in plot_hms)
    vmax = max(np.max(heatmaps[hm_name]) for hm_name in plot_hms)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5.8,2.3))
    for heatmap_name, ax, ax_title in zip(plot_hms, axes, plot_titles):
        heatmap = heatmaps[heatmap_name]
        ims = ax.imshow(heatmap, cmap="Reds", vmin=vmin, vmax=vmax)

        # Configure axis
        ax.set_yticks(np.arange(len(ds_list)))
        ax.set_yticklabels(np.arange(len(ds_list)) + 1)

        ax.set_xticks(np.arange(MAX_LAYERS))
        ax.set_xticklabels(np.arange(MAX_LAYERS) + 1)
        ax.set_title(ax_title)
        ax.set_xlabel(r"Model $L$")

    axes[0].set_ylabel(r"True $L$")

    # Colorbar
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8),
    cb_ax = fig.add_axes([0.83, 0.16, 0.02, 0.68])
    fig.colorbar(ims, cax=cb_ax)

    # Save pdf
    save_path = os.path.join(OUTPUT_DIR, "{}_heatmap.pdf".format(title, heatmap_name))
    plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

line_metric_is = {
    "mean-mae": 1,
    "std-mae": 2,
    "rmse": 4,
    "crps": 5,
}

# Make line plots
for name, ds in LINE_DS.items():
    #Predictions from true posterior
    post_pred = get_true_pred(ds)

    for metric, metric_i in line_metric_is.items():
        # Model predictions
        mean_res = np.array([ds_res[ds][l]["mean"][metric_i] for l in range(MAX_LAYERS)])
        std_res = np.array([ds_res[ds][l]["std"][metric_i] for l in range(MAX_LAYERS)])

        fig, ax = plt.subplots(figsize=(4,2.5))
        xs = np.arange(MAX_LAYERS) + 1

        # Plot lines
        ax.plot(xs, mean_res, lw=3, c=LINE_COLOR, label="DGMRF")

        ci = std_res*1.96
        ax.fill_between(xs, mean_res - ci, mean_res + ci, alpha=0.2, color=LINE_COLOR)

        ax.axhline(y=post_pred[metric], lw=3, color="red", label="True posterior",
                ls="dashed")

        # Legend
        ax.legend()

        # Configure axis
        ax.set_ylabel(metric.upper())

        ax.set_xlabel(r"Layers $L$")
        ax.set_xticks(xs)
        ax.set_xlim(1,MAX_LAYERS)
        #ax.set_ylim(0.,np.max(mean_res + ci))
        ax.set_xticklabels(xs)
        ax.ticklabel_format(axis="y", scilimits=(2,-1), useMathText=True)

        subtitle = str.split(metric, "_")[0]

        # Save pdf
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "{}_{}.pdf".format(name, subtitle))
        plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0.025)
        plt.close

