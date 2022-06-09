import networkx as nx
import torch_geometric as ptg
import os
import wandb
import numpy as np

import matplotlib.pyplot as plt

# Saves plot both as pdf and image on wandb
def save_plot(fig, name, wandb_name=None):
    # Save .pdf-file
    if wandb.config.save_pdf:
        save_path = os.path.join(wandb.run.dir, "{}.pdf".format(name))
        plt.savefig(save_path)

    # Save image to wandb
    if not wandb_name:
        wandb_name = name
    wandb.log({wandb_name: wandb.Image(plt)})
    plt.close(fig)

def plot_graph(graph, name, show=False, return_plot=False, title=None,
        zoom=None, node_size=30):
    # Perform all computation on cpu (for plotting)
    graph = graph.clone().to(device="cpu")
    n_nodes = graph.num_nodes

    graph_networkx = ptg.utils.to_networkx(graph, to_undirected=True)
    plot_pos = graph.pos.cpu().numpy()

    node_color = graph.x.flatten()

    fig, ax = plt.subplots(figsize=(15,12))
    # Determine node size for plots
    if n_nodes > 3000:
        node_size = 10
    if n_nodes > 10000:
        node_size = 5

    nodes = nx.draw_networkx_nodes(graph_networkx, pos=plot_pos, node_size=node_size,
            node_color=node_color)
    nx.draw_networkx_edges(graph_networkx, pos=plot_pos, width=0.5)

    # Draw masked area
    if hasattr(graph, "mask_limits"):
        masks_cpu = graph.mask_limits.cpu()
        if masks_cpu.dim() == 2:
            # If only single mask, expand to array of 1 entry
            masks_cpu = masks_cpu.unsqueeze(0)

        for mask in masks_cpu:
            mask_width, mask_height = mask[1] - mask[0]
            ax.add_patch(plt.Rectangle(mask[0], mask_width, mask_height,
                                fill=False, linewidth=2, edgecolor=(1.,0.,0.,0.7)))

    if type(zoom) == np.ndarray: # Really only good way to check if zoom is given
        ax.set_xlim(*zoom[:,0])
        ax.set_ylim(*zoom[:,1])

    plt.colorbar(nodes)

    if title:
        plt.title(title, size=15)

    if return_plot:
        return fig

    if show:
        plt.show()
    else:
        save_plot(fig, name)

