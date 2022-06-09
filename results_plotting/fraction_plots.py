import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import os
import sys
sys.path.insert(0,'..')

import utils

plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

DATA_DESC = [
    (
        "mix",
        "gmrf_prec_mix32_random",
        "mix_fraction_dgmrf.csv",
        "mix_fraction_baselines.csv",
        "dgp_mix_res.csv",
        ("upper right", None),
        ((1.45e-2, 2.65e-2), (0.8e-2, 1.82e-2)),
    ),
    (
        "croc",
        "wiki_crocodile_random",
        "crocodile_fraction_dgmrf.csv",
        "crocodile_fraction_baselines.csv",
        "dgp_croc_res.csv",
        ("center right", (1.0,0.64)),
        ((1.05, 2.3), (0.55, 1.4)),
    ),
]

LABEL_FRACTIONS = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
OUTPUT_DIR = "figures"

LINE_COLORS = ("purple", "limegreen", "red", "orange")
LINE_STYLES = ("solid", "dotted", "dashed", "dashdot")
STYLE_OFFSET = 2 # Start index for baselines
LINE_WIDTH = 2
CONF_ALPHA = 0.15

METRICS = {
    "RMSE": 0,
    "CRPS": 1,
}

# Baselines to include
BASELINE_FILTER = ("Graph GP", "IGMRF")

for (short_name, ds_name, dgmrf_file, baseline_file, dgp_file,
        legend_pos_config, y_lims) in DATA_DESC:
    # DGMRF
    df = pd.read_csv(dgmrf_file)
    ds_list = df["dataset"].to_numpy()
    res_arr = df.to_numpy()[:,3:].astype(float)

    all_ds = np.unique(ds_list)
    ds_res = {}

    for ds in all_ds:
        # Extract rows associated with dataset
        res_rows = res_arr[ds_list == ds]
        ds_res[ds] = {
            "mean": np.mean(res_rows, axis=0),
            "std": np.std(res_rows, axis=0, ddof=1), # Unbiased estimate
        }

    fraction_res = [ds_res[ds_name + "_" + str(fraction)] for fraction
            in LABEL_FRACTIONS]

    # DGP
    dgp_df = pd.read_csv(dgp_file)
    dgp_df_np = dgp_df.to_numpy()
    dgp_fractions = dgp_df_np[:,0]
    dgp_fraction_means = np.array([np.mean(dgp_df_np[dgp_fractions == fraction, 2:],
        axis=0) for fraction in LABEL_FRACTIONS])
    dgp_fraction_std = np.array([np.std(dgp_df_np[dgp_fractions == fraction, 2:],
        axis=0, ddof=1) for fraction in LABEL_FRACTIONS])
    dgp_res = {
        "mean": dgp_fraction_means[:, 1:3], # RMSE, CRPS columns"
        "std": dgp_fraction_std[:, 1:3],
    }

    # Other baselines
    baseline_df = pd.read_csv(baseline_file)
    bl_filter = baseline_df["Baselines"].isin(BASELINE_FILTER)
    baseline_df = baseline_df[bl_filter] # Filter out some baselines
    bl_names = baseline_df["Baselines"].to_numpy()
    bl_metrics = (
        baseline_df.to_numpy()[:, 1:7].astype(float), # RMSE
        baseline_df.to_numpy()[:, 7:].astype(float), # CRPS
    )

    for metric_name, metric_i in METRICS.items():
        metric_mean = np.array([f["mean"][metric_i] for f in fraction_res])
        metric_std = np.array([f["std"][metric_i] for f in fraction_res])

        fig, ax = plt.subplots(figsize=(4,2.7))

        # Use % of labels as x-label
        xs = 100.*(1-LABEL_FRACTIONS)

        # Plot DGMRF lines
        ax.plot(xs, metric_mean, lw=LINE_WIDTH, c=LINE_COLORS[0],
                label="DGMRF", ls=LINE_STYLES[0])

        ci = metric_std*1.96
        ax.fill_between(xs, metric_mean - ci, metric_mean + ci, alpha=CONF_ALPHA,
                color=LINE_COLORS[0])

        # Plot DGP lines
        dgp_metric_mean = dgp_res["mean"][:, metric_i]
        dgp_metric_std = dgp_res["std"][:, metric_i]
        ax.plot(xs, dgp_metric_mean, lw=LINE_WIDTH, c=LINE_COLORS[1],
                label=r'DGP ($\times 10$)', ls=LINE_STYLES[1])

        ci = dgp_metric_std*1.96
        ax.fill_between(xs, dgp_metric_mean - ci, dgp_metric_mean + ci, alpha=CONF_ALPHA,
                color=LINE_COLORS[1])

        # Plot baselines
        for bl_name, bl_res, col, line_style in zip(bl_names, bl_metrics[metric_i],
                LINE_COLORS[STYLE_OFFSET:], LINE_STYLES[STYLE_OFFSET:]):
            ax.plot(xs, bl_res, lw=LINE_WIDTH, label=r'{}'.format(bl_name),
                    c=col, ls=line_style)

        # Legend
        legend_pos, legend_bbox = legend_pos_config
        ax.legend(loc=legend_pos, handlelength=3, bbox_to_anchor=legend_bbox,
                prop={'size': 8})

        # Configure axis
        ax.set_ylabel(metric_name.upper())

        ax.set_xlabel("Observed nodes (\%)")
        ax.set_xticks(xs)
        ax.set_xlim(0.,100.)
        ax.set_ylim(*y_lims[metric_i])
        ax.set_xticklabels(xs)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.ticklabel_format(axis="y", scilimits=(-1,3), useMathText=True)

        save_name = "fraction_{}_{}".format(short_name, metric_name.lower())

        # Save pdf
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "{}.pdf".format(save_name))
        plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0.025)
        plt.close()
        png_save_path = os.path.join(OUTPUT_DIR, "{}.png".format(save_name))
        plt.savefig(png_save_path, bbox_inches = 'tight', pad_inches = 0)
        plt.close()

