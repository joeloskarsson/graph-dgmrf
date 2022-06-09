import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

OUTPUT_DIR = "figures"

LINE_COLORS = ("purple", "limegreen", "red", "orange", "navy")
LINE_STYLES = ("dotted", "dashdot", (0, (3, 1, 1, 1, 1, 1)), "solid", "dashed")
STYLE_OFFSET = 2 # Start index for baselines
LINE_WIDTH = 2

MAX_K = 100
q_values = (0.99, 0.95, 0.9, 0.5, 0.1)

plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

fig, ax = plt.subplots(figsize=(6,3))
ks = np.arange(1, MAX_K+1)

label_template = "$\\left|\\frac{\\beta_l}{\\alpha_l}\\right|$"

for i, q in enumerate(q_values):
    log_part = -1*np.log(1. - q)
    series = (1./ks)*np.power(q, ks)
    sum_part = -1.*np.cumsum(series)
    bound = log_part + sum_part


    ax.plot(ks, bound, lw=LINE_WIDTH, c=LINE_COLORS[i],
        label=r"{} = {}".format(label_template, q), ls=LINE_STYLES[i])

# Legend
ax.legend(handlelength=6)

# Configure axis
ax.set_ylabel(r"$\frac{E_K}{N} \leq$")

ax.set_xlabel(r"$K$")
ax.set_xlim(1,MAX_K)
ax.set_ylim(-0.02)

# Save pdf
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "power_series_bound.pdf")
plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0.)
plt.close()

